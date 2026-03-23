from torch import nn
import torch
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    """
    Apply a module on the last dimension while merging preceding dims if needed.
    """
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y


class GLU(nn.Module):
    """
    Gated Linear Unit
    """
    def __init__(self, input_size):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.act = nn.Hardswish()

    def forward(self, x):
        gate = self.act(self.fc1(x))
        value = self.fc2(x)
        return gate * value


class GatedResidualNetwork(nn.Module):
    """
    Lightweight GRN:
        x -> fc1 -> ELU -> fc2 -> dropout -> GLU -> residual -> LayerNorm
    """
    def __init__(
        self,
        input_size,
        hidden_state_size,
        output_size,
        dropout,
        hidden_context_size=None,
        batch_first=False
    ):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size = hidden_state_size
        self.dropout_rate = dropout
        self.batch_first = batch_first

        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(
                nn.Linear(self.input_size, self.output_size),
                batch_first=batch_first
            )

        self.fc1 = TimeDistributed(
            nn.Linear(self.input_size, self.hidden_state_size),
            batch_first=batch_first
        )
        self.elu1 = nn.ELU()

        if self.hidden_context_size is not None:
            self.context = TimeDistributed(
                nn.Linear(self.hidden_context_size, self.hidden_state_size),
                batch_first=batch_first
            )

        self.fc2 = TimeDistributed(
            nn.Linear(self.hidden_state_size, self.output_size),
            batch_first=batch_first
        )

        self.dropout = nn.Dropout(self.dropout_rate)
        self.gate = TimeDistributed(
            GLU(self.output_size),
            batch_first=batch_first
        )
        self.norm = TimeDistributed(
            nn.LayerNorm(self.output_size),
            batch_first=batch_first
        )

    def forward(self, x, context=None):
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        out = self.fc1(x)
        if context is not None:
            context = self.context(context)
            out = out + context

        out = self.elu1(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.gate(out)
        out = out + residual
        out = self.norm(out)
        return out


class LocalSmoothingBranch(nn.Module):
    """
    Local-smoothed temporal branch.

    Input:
        x_var: [B, N, L]

    Output:
        h_local: [B, N, L]

    Idea:
        Use depthwise 1D conv along temporal dimension to capture local smoothed patterns,
        then refine with a lightweight GRN.
    """
    def __init__(self, seq_len, hidden_dim=64, dropout=0.1, kernel_size=5):
        super(LocalSmoothingBranch, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        padding = kernel_size // 2

        # depthwise conv over time for each variable independently
        self.local_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            bias=True
        )

        self.local_grn = GatedResidualNetwork(
            input_size=seq_len,
            hidden_state_size=hidden_dim,
            output_size=seq_len,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x_var):
        """
        x_var: [B, N, L]
        """
        B, N, L = x_var.shape
        x_reshape = x_var.contiguous().view(B * N, 1, L)   # [B*N, 1, L]
        x_local = self.local_conv(x_reshape)               # [B*N, 1, L]
        x_local = x_local.view(B, N, L)                    # [B, N, L]
        h_local = self.local_grn(x_local)                  # [B, N, L]
        return h_local


class CoarseScaleBranch(nn.Module):
    """
    Coarse-scale temporal branch.

    Input:
        x_var: [B, N, L]

    Output:
        h_coarse: [B, N, L]

    Idea:
        Build a coarse temporal representation by downsampling over time,
        then up-project back to length L and refine by GRN.
    """
    def __init__(self, seq_len, hidden_dim=64, dropout=0.1, coarse_ratio=4):
        super(CoarseScaleBranch, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.coarse_ratio = max(1, coarse_ratio)

        coarse_len = max(1, seq_len // self.coarse_ratio)
        self.coarse_len = coarse_len

        self.pool = nn.AdaptiveAvgPool1d(coarse_len)
        self.up_proj = TimeDistributed(
            nn.Linear(coarse_len, seq_len),
            batch_first=True
        )

        self.coarse_grn = GatedResidualNetwork(
            input_size=seq_len,
            hidden_state_size=hidden_dim,
            output_size=seq_len,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x_var):
        """
        x_var: [B, N, L]
        """
        B, N, L = x_var.shape
        x_reshape = x_var.contiguous().view(B * N, 1, L)   # [B*N, 1, L]
        x_coarse = self.pool(x_reshape).squeeze(1)         # [B*N, coarse_len]
        x_coarse = x_coarse.view(B, N, self.coarse_len)    # [B, N, coarse_len]
        x_up = self.up_proj(x_coarse)                      # [B, N, L]
        h_coarse = self.coarse_grn(x_up)                   # [B, N, L]
        return h_coarse


class MultiScaleTemporalScoring(nn.Module):
    """
    Multi-scale temporal scoring network.

    Input:
        x_var: [B, N, L]

    Output:
        logits: [B, N, L]

    Branches:
        1) raw-scale branch
        2) local-smoothed branch
        3) coarse-scale branch

    Fusion:
        Per-variable scale gating over branch representations.
    """
    def __init__(
        self,
        seq_len,
        hidden_dim=64,
        dropout=0.1,
        local_kernel_size=5,
        coarse_ratio=4
    ):
        super(MultiScaleTemporalScoring, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # raw branch
        self.raw_grn = GatedResidualNetwork(
            input_size=seq_len,
            hidden_state_size=hidden_dim,
            output_size=seq_len,
            dropout=dropout,
            batch_first=True
        )

        # local branch
        self.local_branch = LocalSmoothingBranch(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=local_kernel_size
        )

        # coarse branch
        self.coarse_branch = CoarseScaleBranch(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            coarse_ratio=coarse_ratio
        )

        # branch-wise scale gate
        # Use global summary over time for each branch representation:
        # [B,N,L] -> [B,N,1], then concat 3 branches -> [B,N,3]
        self.scale_gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )

        # final fusion refinement + output projection
        self.fusion_grn = GatedResidualNetwork(
            input_size=seq_len,
            hidden_state_size=hidden_dim,
            output_size=seq_len,
            dropout=dropout,
            batch_first=True
        )

        self.out_proj = TimeDistributed(
            nn.Linear(seq_len, seq_len),
            batch_first=True
        )

    def forward(self, x_var):
        """
        x_var: [B, N, L]
        """
        # multi-scale branch features
        h_raw = self.raw_grn(x_var)              # [B, N, L]
        h_local = self.local_branch(x_var)       # [B, N, L]
        h_coarse = self.coarse_branch(x_var)     # [B, N, L]

        # summaries for scale gating
        s_raw = h_raw.mean(dim=-1, keepdim=True)        # [B, N, 1]
        s_local = h_local.mean(dim=-1, keepdim=True)    # [B, N, 1]
        s_coarse = h_coarse.mean(dim=-1, keepdim=True)  # [B, N, 1]

        scale_summary = torch.cat([s_raw, s_local, s_coarse], dim=-1)  # [B, N, 3]
        scale_logits = self.scale_gate(scale_summary)                   # [B, N, 3]
        scale_weights = torch.softmax(scale_logits, dim=-1)             # [B, N, 3]

        w_raw = scale_weights[..., 0].unsqueeze(-1)     # [B, N, 1]
        w_local = scale_weights[..., 1].unsqueeze(-1)   # [B, N, 1]
        w_coarse = scale_weights[..., 2].unsqueeze(-1)  # [B, N, 1]

        # gated fusion
        h_fused = w_raw * h_raw + w_local * h_local + w_coarse * h_coarse  # [B, N, L]

        # fusion refinement + final logits
        h_fused = self.fusion_grn(h_fused)      # [B, N, L]
        logits = self.out_proj(h_fused)         # [B, N, L]
        return logits


class TemporalInputSelection(nn.Module):
    """
    Multi-scale temporal input selection for iTransformer compatibility.

    Input:
        x: [B, L, N]

    Output dict:
        {
            'selected_x': [B, L, N],
            'logits':     [B, N, L],
            'weights':    [B, N, L],
            'agg_repr':   [B, N],
            'sparse_loss': scalar
        }

    Key design:
        1) Multi-scale temporal scoring:
            - raw scale
            - local smoothed scale
            - coarse scale
        2) Final selected_x uses residual modulation for stable integration
           with iTransformer backbone.
    """

    def __init__(
        self,
        seq_len,
        hidden_dim=64,
        dropout=0.1,
        temperature=1.0,
        sparse_type='entropy',
        eps=1e-8,
        residual_scale=0.5,
        local_kernel_size=5,
        coarse_ratio=4
    ):
        super(TemporalInputSelection, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = temperature
        self.sparse_type = sparse_type
        self.eps = eps
        self.residual_scale = residual_scale

        self.temporal_scorer = MultiScaleTemporalScoring(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            local_kernel_size=local_kernel_size,
            coarse_ratio=coarse_ratio
        )

    def _compute_sparse_loss(self, weights_bnL, logits_bnL):
        """
        weights_bnL: [B, N, L]
        logits_bnL:  [B, N, L]
        """
        if self.sparse_type == 'entropy':
            entropy = -(weights_bnL * torch.log(weights_bnL + self.eps)).sum(dim=-1)  # [B, N]
            sparse_loss = entropy.mean()
        elif self.sparse_type == 'logits_l1':
            sparse_loss = torch.mean(torch.abs(logits_bnL))
        elif self.sparse_type == 'none':
            sparse_loss = weights_bnL.new_tensor(0.0)
        else:
            raise ValueError(f"Unsupported sparse_type: {self.sparse_type}")
        return sparse_loss

    def forward(self, x):
        """
        x: [B, L, N]
        """
        # [B, L, N] -> [B, N, L]
        x_var = x.permute(0, 2, 1)

        # multi-scale logits over temporal dimension
        logits = self.temporal_scorer(x_var)  # [B, N, L]

        # temperature-scaled softmax over time dimension
        weights = torch.softmax(logits / self.temperature, dim=-1)  # [B, N, L]

        # aggregated representation for compatibility / controller state
        agg_repr = torch.sum(weights * x_var, dim=-1)  # [B, N]

        # residual modulation on original input
        modulation = 1.0 + self.residual_scale * weights
        selected_x_var = x_var * modulation
        selected_x = selected_x_var.permute(0, 2, 1)  # [B, L, N]

        sparse_loss = self._compute_sparse_loss(weights, logits)

        return {
            'selected_x': selected_x,   # [B, L, N]
            'logits': logits,           # [B, N, L]
            'weights': weights,         # [B, N, L]
            'agg_repr': agg_repr,       # [B, N]
            'sparse_loss': sparse_loss
        }
