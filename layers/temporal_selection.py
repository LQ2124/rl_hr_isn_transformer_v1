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


class LightweightTemporalScoring(nn.Module):
    """
    Lightweight temporal scoring network.

    Input:
        x_var: [B, N, L]

    Output:
        logits: [B, N, L]

    Idea:
        For each variable, score its entire history over time using a light GRN/MLP.
        No heavy input reconstruction.
    """
    def __init__(self, seq_len, hidden_dim=64, dropout=0.1):
        super(LightweightTemporalScoring, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.temporal_grn = GatedResidualNetwork(
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
        h = self.temporal_grn(x_var)   # [B, N, L]
        logits = self.out_proj(h)      # [B, N, L]
        return logits


class TemporalInputSelection(nn.Module):
    """
    Simplified temporal input selection for iTransformer compatibility.

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

    Key changes vs old version:
        1) Remove heavy MyNet reconstruction.
        2) Directly generate temporal logits from raw input.
        3) selected_x is residual modulation of original x, not reconstructed x.
    """

    def __init__(
        self,
        seq_len,
        hidden_dim=64,
        dropout=0.1,
        temperature=1.0,
        sparse_type='entropy',
        eps=1e-8,
        residual_scale=0.5
    ):
        super(TemporalInputSelection, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = temperature
        self.sparse_type = sparse_type
        self.eps = eps
        self.residual_scale = residual_scale

        self.temporal_scorer = LightweightTemporalScoring(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            dropout=dropout
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

        # lightweight logits over temporal dimension
        logits = self.temporal_scorer(x_var)  # [B, N, L]

        # temperature-scaled softmax over time dimension
        weights = torch.softmax(logits / self.temperature, dim=-1)  # [B, N, L]

        # aggregated representation for compatibility / controller state
        agg_repr = torch.sum(weights * x_var, dim=-1)  # [B, N]

        # residual modulation on original input
        # selected_x_var = x_var * (1 + residual_scale * weights)
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
