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

    def __init__(self, eps=1e-8, target_index=-1):
        super(SelectionStateBuilder, self).__init__()
        self.eps = eps
        self.target_index = target_index

        # original 11 + target-aware 14 = 25
        self.state_dim = 25

    def _get_target_feature(self, x_bn):
        """
        x_bn: [B, N]
        returns target slice [B, 1]
        """
        return x_bn[:, self.target_index:self.target_index + 1] if self.target_index != -1 else x_bn[:, -1:]

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

        # --------------------------------------------------------------
        # target-aware features
        # --------------------------------------------------------------
        target_raw_summary = self._get_target_feature(s_raw)          # [B, 1]
        target_local_summary = self._get_target_feature(s_local)      # [B, 1]
        target_coarse_summary = self._get_target_feature(s_coarse)    # [B, 1]

        target_base_entropy = self._get_target_feature(base_entropy)  # [B, 1]

        target_scale_raw = self._get_target_feature(scale_raw)        # [B, 1]
        target_scale_local = self._get_target_feature(scale_local)    # [B, 1]
        target_scale_coarse = self._get_target_feature(scale_coarse)  # [B, 1]

        target_raw_local_gap = self._get_target_feature(raw_local_gap)        # [B, 1]
        target_raw_coarse_gap = self._get_target_feature(raw_coarse_gap)      # [B, 1]
        target_local_coarse_gap = self._get_target_feature(local_coarse_gap)  # [B, 1]

        # --------------------------------------------------------------
        # target-aware regime-like proxies
        # --------------------------------------------------------------
        target_mean_like = (
            target_scale_raw * target_raw_summary +
            target_scale_local * target_local_summary +
            target_scale_coarse * target_coarse_summary
        )  # [B, 1]

        target_std_like = torch.sqrt(
            (
                (target_raw_summary - target_mean_like) ** 2 +
                (target_local_summary - target_mean_like) ** 2 +
                (target_coarse_summary - target_mean_like) ** 2
            ) / 3.0 + self.eps
        )  # [B, 1]

        target_recent_slope_like = target_raw_summary - target_coarse_summary  # [B, 1]

        target_scale_range = torch.max(
            torch.cat([target_scale_raw, target_scale_local, target_scale_coarse], dim=-1),
            dim=-1,
            keepdim=True
        )[0] - torch.min(
            torch.cat([target_scale_raw, target_scale_local, target_scale_coarse], dim=-1),
            dim=-1,
            keepdim=True
        )[0]  # [B, 1]

        B, N = agg_repr.shape

        target_raw_summary_expand = target_raw_summary.expand(B, N)
        target_local_summary_expand = target_local_summary.expand(B, N)
        target_coarse_summary_expand = target_coarse_summary.expand(B, N)

        target_base_entropy_expand = target_base_entropy.expand(B, N)

        target_scale_raw_expand = target_scale_raw.expand(B, N)
        target_scale_local_expand = target_scale_local.expand(B, N)
        target_scale_coarse_expand = target_scale_coarse.expand(B, N)

        target_mean_like_expand = target_mean_like.expand(B, N)
        target_std_like_expand = target_std_like.expand(B, N)
        target_recent_slope_like_expand = target_recent_slope_like.expand(B, N)
        target_scale_range_expand = target_scale_range.expand(B, N)

        target_raw_local_gap_expand = target_raw_local_gap.expand(B, N)
        target_raw_coarse_gap_expand = target_raw_coarse_gap.expand(B, N)
        target_local_coarse_gap_expand = target_local_coarse_gap.expand(B, N)

        state = torch.stack(
            [
                # original state
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
                scale_coarse,

                # target-aware summaries
                target_raw_summary_expand,
                target_local_summary_expand,
                target_coarse_summary_expand,
                target_base_entropy_expand,
                target_scale_raw_expand,
                target_scale_local_expand,
                target_scale_coarse_expand,

                # target regime proxies
                target_mean_like_expand,
                target_std_like_expand,
                target_recent_slope_like_expand,
                target_scale_range_expand,

                # target branch disagreement
                target_raw_local_gap_expand,
                target_raw_coarse_gap_expand,
                target_local_coarse_gap_expand
            ],
            dim=-1
        )  # [B, N, 25]

        return state


class ScalePreferenceActor(nn.Module):
    """
    Stochastic actor for residual-gated dynamic scale preference control.

    Input:
        state: [B, N, state_dim]

    Outputs:
        action_mean:     [B, N, 3]
        action_std:      [B, N, 3]
        action_delta:    [B, N, 3]
        gate_alpha:      [B, N, 1]   in [0, 1]
        log_prob:        [B, N]
        entropy:         [B, N]
        action_strength: [B, N]
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

        # stochastic action
        self.mean_head = nn.Linear(hidden_dim, 3)
        self.log_std_head = nn.Linear(hidden_dim, 3)

        # residual gate
        self.gate_head = nn.Linear(hidden_dim, 1)

        # bounded stochasticity for stable exploration
        self.min_log_std = -5.0
        self.max_log_std = -1.0

    def forward(self, state, deterministic=False):
        """
        state: [B, N, state_dim]
        deterministic:
            - False during training: sample stochastic action
            - True during evaluation: use mean action
        """
        h = self.backbone(state)

        action_mean_raw = self.mean_head(h)          # [B, N, 3]
        action_log_std = self.log_std_head(h)        # [B, N, 3]
        action_log_std = torch.clamp(action_log_std, self.min_log_std, self.max_log_std)
        action_std = torch.exp(action_log_std)       # [B, N, 3]

        dist = Normal(action_mean_raw, action_std)

        if deterministic:
            sampled_raw = action_mean_raw
        else:
            sampled_raw = dist.rsample()

        # bounded residual correction
        action_delta = self.action_scale * torch.tanh(sampled_raw)  # [B, N, 3]

        # residual gate in [0,1]
        gate_alpha = torch.sigmoid(self.gate_head(h))  # [B, N, 1]

        # policy quantities
        log_prob = dist.log_prob(sampled_raw).sum(dim=-1)  # [B, N]
        entropy = dist.entropy().sum(dim=-1)               # [B, N]

        action_strength = torch.mean(torch.abs(action_delta), dim=-1)  # [B, N]

        return {
            'action_mean': action_mean_raw,
            'action_std': action_std,
            'action_delta': action_delta,
            'gate_alpha': gate_alpha,
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

    Residual-gated design:
        dynamic scale preference is a gated residual correction over static scale preference:
            S_dyn = Softmax( log(S_base + eps) + gate_alpha * action_delta )

    This makes the static selector the main path and dynamic RL only performs
    conservative residual calibration.
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

        self.state_builder = SelectionStateBuilder(eps=eps, target_index=-1)

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
        gate_alpha = actor_out['gate_alpha']            # [B, N, 1]
        log_prob = actor_out['log_prob']                # [B, N]
        entropy = actor_out['entropy']                  # [B, N]
        action_strength = actor_out['action_strength']  # [B, N]
        action_mean = actor_out['action_mean']          # [B, N, 3]
        action_std = actor_out['action_std']            # [B, N, 3]

        value = self.critic(state)                      # [B, N, 1]

        # -------------------------------------------------------------
        # Residual-gated dynamic scale preference
        # S_dyn = Softmax( log(S_base + eps) + gate_alpha * action_delta )
        # -------------------------------------------------------------
        static_scale_logits = torch.log(static_scale_weights + self.eps)      # [B, N, 3]
        gated_residual = gate_alpha * action_delta                            # [B, N, 3]
        dynamic_scale_logits = static_scale_logits + gated_residual           # [B, N, 3]
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
            'gate_alpha': gate_alpha,                    # [B, N, 1]
            'log_prob': log_prob,                        # [B, N]
            'entropy': entropy,                          # [B, N]
            'action_strength': action_strength,          # [B, N]

            # residual-gated dynamic scale preference
            'dynamic_scale_logits': dynamic_scale_logits,      # [B, N, 3]
            'dynamic_scale_weights': dynamic_scale_weights,    # [B, N, 3]

            # dynamic temporal selection
            'calibrated_logits': calibrated_logits,      # [B, N, L]
            'dynamic_weights': dynamic_weights,          # [B, N, L]
            'selected_x': selected_x                     # [B, L, N]
        }
