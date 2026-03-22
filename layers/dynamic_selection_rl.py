import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectionStateBuilder(nn.Module):
    """
    Build lightweight per-variable state for dynamic temporal selection.

    Inputs:
        base_logits:  [B, N, L]
        base_weights: [B, N, L]
        agg_repr:     [B, N]

    Output:
        state: [B, N, state_dim]
    """

    def __init__(self, eps=1e-8):
        super(SelectionStateBuilder, self).__init__()
        self.eps = eps
        self.state_dim = 4  # agg_repr, logits_mean, logits_std, weights_entropy

    def forward(self, base_logits, base_weights, agg_repr):
        logits_mean = base_logits.mean(dim=-1)  # [B, N]
        logits_std = base_logits.std(dim=-1, unbiased=False)  # [B, N]
        weights_entropy = -(base_weights * torch.log(base_weights + self.eps)).sum(dim=-1)  # [B, N]

        state = torch.stack(
            [agg_repr, logits_mean, logits_std, weights_entropy],
            dim=-1
        )  # [B, N, 4]

        return state


class SelectionActor(nn.Module):
    """
    Lightweight actor:
        state -> alpha, beta

    Input:
        state: [B, N, state_dim]

    Output:
        alpha: [B, N, 1]
        beta:  [B, N, 1]
    """

    def __init__(self, state_dim, hidden_dim=64, alpha_scale=0.1, beta_scale=0.1):
        super(SelectionActor, self).__init__()
        self.alpha_scale = alpha_scale
        self.beta_scale = beta_scale

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        state: [B, N, state_dim]
        """
        h = self.backbone(state)

        alpha_raw = self.alpha_head(h)  # [B, N, 1]
        beta_raw = self.beta_head(h)    # [B, N, 1]

        # Stable bounded actions:
        # alpha around 1, beta around 0
        alpha = 1.0 + self.alpha_scale * torch.tanh(alpha_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)

        return alpha, beta


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
    Dynamic controller for static temporal selection refinement.

    Workflow:
        base_logits z
            -> state builder
            -> actor(alpha, beta)
            -> critic(value)
            -> calibrated_logits zhat = alpha * z + beta
            -> dynamic_weights = softmax(zhat)
            -> selected_x_dynamic

    Inputs:
        x:            [B, L, N]
        base_logits:  [B, N, L]
        base_weights: [B, N, L]
        agg_repr:     [B, N]

    Outputs:
        dict with:
            state:             [B, N, state_dim]
            alpha:             [B, N, 1]
            beta:              [B, N, 1]
            value:             [B, N, 1]
            calibrated_logits: [B, N, L]
            dynamic_weights:   [B, N, L]
            selected_x:        [B, L, N]
    """

    def __init__(
        self,
        seq_len,
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        alpha_scale=0.1,
        beta_scale=0.1,
        eps=1e-8
    ):
        super(DynamicSelectionController, self).__init__()
        self.seq_len = seq_len
        self.eps = eps

        self.state_builder = SelectionStateBuilder(eps=eps)

        self.actor = SelectionActor(
            state_dim=self.state_builder.state_dim,
            hidden_dim=actor_hidden_dim,
            alpha_scale=alpha_scale,
            beta_scale=beta_scale
        )

        self.critic = SelectionCritic(
            state_dim=self.state_builder.state_dim,
            hidden_dim=critic_hidden_dim
        )

    def calibrate_logits(self, base_logits, alpha, beta):
        """
        base_logits: [B, N, L]
        alpha:       [B, N, 1]
        beta:        [B, N, 1]
        """
        calibrated_logits = alpha * base_logits + beta
        return calibrated_logits

    def forward(self, x, base_logits, base_weights, agg_repr):
        """
        Args:
            x:            [B, L, N]
            base_logits:  [B, N, L]
            base_weights: [B, N, L]
            agg_repr:     [B, N]

        Returns:
            dict
        """
        state = self.state_builder(base_logits, base_weights, agg_repr)  # [B, N, 4]

        alpha, beta = self.actor(state)          # [B, N, 1], [B, N, 1]
        value = self.critic(state)               # [B, N, 1]

        calibrated_logits = self.calibrate_logits(base_logits, alpha, beta)  # [B, N, L]
        dynamic_weights = torch.softmax(calibrated_logits, dim=-1)           # [B, N, L]

        # x: [B, L, N] -> [B, N, L]
        x_var = x.permute(0, 2, 1)
        selected_x_var = dynamic_weights * x_var
        selected_x = selected_x_var.permute(0, 2, 1)  # [B, L, N]

        return {
            'state': state,
            'alpha': alpha,
            'beta': beta,
            'value': value,
            'calibrated_logits': calibrated_logits,
            'dynamic_weights': dynamic_weights,
            'selected_x': selected_x
        }
