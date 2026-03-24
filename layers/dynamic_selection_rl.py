import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SelectionStateBuilder(nn.Module):
    """
    Build lightweight target-aware multi-scale state for dynamic temporal selection.

    Inputs:
        branch_logits:
            raw    [B, N, L]
            local  [B, N, L]
            coarse [B, N, L]
        base_weights:         [B, N, L]
        agg_repr:             [B, N]
        static_scale_weights: [B, N, 3]
        summaries:
            raw    [B, N]
            local  [B, N]
            coarse [B, N]

    Output:
        state: [B, N, state_dim]
    """

    def __init__(self, eps=1e-8):
        super(SelectionStateBuilder, self).__init__()
        self.eps = eps

        # [agg_repr,
        #  raw_summary, local_summary, coarse_summary,
        #  base_entropy,
        #  raw_local_gap, raw_coarse_gap, local_coarse_gap,
        #  static_scale_raw, static_scale_local, static_scale_coarse]
        self.state_dim = 11

    def forward(
        self,
        branch_logits,
        base_weights,
        agg_repr,
        static_scale_weights,
        summaries
    ):
        z_raw = branch_logits['raw']         # [B, N, L]
        z_local = branch_logits['local']     # [B, N, L]
        z_coarse = branch_logits['coarse']   # [B, N, L]

        s_raw = summaries['raw']             # [B, N]
        s_local = summaries['local']         # [B, N]
        s_coarse = summaries['coarse']       # [B, N]

        base_entropy = -(base_weights * torch.log(base_weights + self.eps)).sum(dim=-1)  # [B, N]

        raw_local_gap = torch.mean(torch.abs(z_raw - z_local), dim=-1)       # [B, N]
        raw_coarse_gap = torch.mean(torch.abs(z_raw - z_coarse), dim=-1)     # [B, N]
        local_coarse_gap = torch.mean(torch.abs(z_local - z_coarse), dim=-1) # [B, N]

        scale_raw = static_scale_weights[..., 0]      # [B, N]
        scale_local = static_scale_weights[..., 1]    # [B, N]
        scale_coarse = static_scale_weights[..., 2]   # [B, N]

        state = torch.stack(
            [
                agg_repr,
                s_raw,
                s_local,
                s_coarse,
                base_entropy,
                raw_local_gap,
                raw_coarse_gap,
                local_coarse_gap,
                scale_raw,
                scale_local,
                scale_coarse
            ],
            dim=-1
        )  # [B, N, 11]

        return state


class ScalePreferenceActor(nn.Module):
    """
    Stochastic actor for dynamic scale preference control.

    Input:
        state: [B, N, state_dim]

    Outputs:
        action_mean:   [B, N, 3]
        action_std:    [B, N, 3]
        action_delta:  [B, N, 3]   (sampled action in bounded space)
        log_prob:      [B, N]
        entropy:       [B, N]
        action_strength:[B, N]
    """

    def __init__(self, state_dim, hidden_dim=64, action_scale=0.5):
        super(ScalePreferenceActor, self).__init__()
        self.action_scale = action_scale

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # mean of action distribution
        self.mean_head = nn.Linear(hidden_dim, 3)

        # state-dependent log-std
        self.log_std_head = nn.Linear(hidden_dim, 3)

        # numerical bounds for stability
        self.min_log_std = -5.0
        self.max_log_std = 1.0

    def forward(self, state, deterministic=False):
        """
        state: [B, N, state_dim]
        deterministic:
            - False during training: sample stochastic action
            - True during evaluation: use mean action
        """
        h = self.backbone(state)

        action_mean_raw = self.mean_head(h)                      # [B, N, 3]
        action_log_std = self.log_std_head(h)                   # [B, N, 3]
        action_log_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std)
        action_std = torch.exp(action_log_std)                  # [B, N, 3]

        dist = Normal(action_mean_raw, action_std)

        if deterministic:
            sampled_raw = action_mean_raw
        else:
            sampled_raw = dist.rsample()  # reparameterized sample

        # bounded action in [-action_scale, action_scale]
        action_delta = self.action_scale * torch.tanh(sampled_raw)   # [B, N, 3]

        # approximate log-prob on pre-tanh action for policy gradient
        # This is a practical approximation and keeps the implementation lightweight.
        # Shape: [B, N, 3] -> sum over action dim => [B, N]
        log_prob = dist.log_prob(sampled_raw).sum(dim=-1)

        # entropy for optional diagnostics / regularization
        entropy = dist.entropy().sum(dim=-1)  # [B, N]

        action_strength = torch.mean(torch.abs(action_delta), dim=-1)  # [B, N]

        return {
            'action_mean': action_mean_raw,   # pre-tanh mean
            'action_std': action_std,
            'action_delta': action_delta,
            'log_prob': log_prob,
            'entropy': entropy,
            'action_strength': action_strength
        }


class SelectionCritic(nn.Module):
    """
    Lightweight critic:
        state -> value estimate

    Input:
        state: [B, N, state_dim]

    Output:
        value: [B, N, 1]
    """

    def __init__(self, state_dim, hidden_dim=64):
        super(SelectionCritic, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.backbone(state)


class DynamicSelectionController(nn.Module):
    """
    RL controller for target-aware multi-scale temporal selection refinement.

    Static selector provides:
        - branch logits: z_raw, z_local, z_coarse
        - static scale fusion weights
        - base weights

    RL controller dynamically adjusts scale preference:
        static scale logits + sampled action_delta -> dynamic scale weights
        dynamic scale weights fuse branch logits -> dynamic logits
        dynamic logits -> dynamic temporal weights -> selected_x_dynamic
    """

    def __init__(
        self,
        seq_len,
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        action_scale=0.5,
        eps=1e-8,
        temperature=1.0,
        residual_scale=0.5
    ):
        super(DynamicSelectionController, self).__init__()
        self.seq_len = seq_len
        self.eps = eps
        self.temperature = temperature
        self.residual_scale = residual_scale

        self.state_builder = SelectionStateBuilder(eps=eps)

        self.actor = ScalePreferenceActor(
            state_dim=self.state_builder.state_dim,
            hidden_dim=actor_hidden_dim,
            action_scale=action_scale
        )

        self.critic = SelectionCritic(
            state_dim=self.state_builder.state_dim,
            hidden_dim=critic_hidden_dim
        )

    def forward(
        self,
        x,
        branch_logits,
        base_weights,
        agg_repr,
        static_scale_weights,
        summaries,
        deterministic=False
    ):
        """
        Args:
            x:                    [B, L, N]
            branch_logits:
                raw/local/coarse  [B, N, L]
            base_weights:         [B, N, L]
            agg_repr:             [B, N]
            static_scale_weights: [B, N, 3]
            summaries:
                raw/local/coarse  [B, N]
            deterministic:
                False in training; True in eval/inference

        Returns:
            dict
        """
        state = self.state_builder(
            branch_logits=branch_logits,
            base_weights=base_weights,
            agg_repr=agg_repr,
            static_scale_weights=static_scale_weights,
            summaries=summaries
        )  # [B, N, state_dim]

        actor_out = self.actor(state, deterministic=deterministic)
        action_delta = actor_out['action_delta']        # [B, N, 3]
        log_prob = actor_out['log_prob']                # [B, N]
        entropy = actor_out['entropy']                  # [B, N]
        action_strength = actor_out['action_strength']  # [B, N]
        action_mean = actor_out['action_mean']          # [B, N, 3]
        action_std = actor_out['action_std']            # [B, N, 3]

        value = self.critic(state)                      # [B, N, 1]

        # convert static scale weights to logits-like domain for additive correction
        static_scale_logits = torch.log(static_scale_weights + self.eps)      # [B, N, 3]
        dynamic_scale_logits = static_scale_logits + action_delta             # [B, N, 3]
        dynamic_scale_weights = torch.softmax(dynamic_scale_logits, dim=-1)   # [B, N, 3]

        w_raw = dynamic_scale_weights[..., 0].unsqueeze(-1)      # [B, N, 1]
        w_local = dynamic_scale_weights[..., 1].unsqueeze(-1)    # [B, N, 1]
        w_coarse = dynamic_scale_weights[..., 2].unsqueeze(-1)   # [B, N, 1]

        z_raw = branch_logits['raw']         # [B, N, L]
        z_local = branch_logits['local']     # [B, N, L]
        z_coarse = branch_logits['coarse']   # [B, N, L]

        calibrated_logits = w_raw * z_raw + w_local * z_local + w_coarse * z_coarse  # [B, N, L]
        dynamic_weights = torch.softmax(calibrated_logits / self.temperature, dim=-1) # [B, N, L]

        x_var = x.permute(0, 2, 1)  # [B, N, L]
        modulation = 1.0 + self.residual_scale * dynamic_weights
        selected_x_var = x_var * modulation
        selected_x = selected_x_var.permute(0, 2, 1)  # [B, L, N]

        return {
            'state': state,
            'value': value,

            # actor outputs
            'action_mean': action_mean,                  # [B, N, 3]
            'action_std': action_std,                    # [B, N, 3]
            'action_delta': action_delta,                # [B, N, 3]
            'log_prob': log_prob,                        # [B, N]
            'entropy': entropy,                          # [B, N]
            'action_strength': action_strength,          # [B, N]

            # dynamic scale preference
            'dynamic_scale_logits': dynamic_scale_logits,      # [B, N, 3]
            'dynamic_scale_weights': dynamic_scale_weights,    # [B, N, 3]

            # dynamic temporal selection
            'calibrated_logits': calibrated_logits,      # [B, N, L]
            'dynamic_weights': dynamic_weights,          # [B, N, L]
            'selected_x': selected_x                     # [B, L, N]
        }
