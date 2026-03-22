from torch import nn
import torch
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    """
    Takes any module and stacks the time dimension with the batch dimension
    before applying the module.
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
        sig = self.act(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    """
    GRN aligned with your provided code:
        primary input
        -> dense
        -> ELU
        -> dense
        -> dropout
        -> gate / GLU
        -> residual connection
        -> add & norm
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
        self.bn = TimeDistributed(
            nn.LayerNorm(self.output_size),
            batch_first=batch_first
        )
        self.gate = TimeDistributed(
            GLU(self.output_size),
            batch_first=batch_first
        )

    def forward(self, x, context=None):
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.bn(x)

        return x


class MyNet(nn.Module):
    """
    Input transformation module aligned with your provided code.

    Original code fixes seq_len=96 and internal expanded feature size=16*num_inputs.
    Here seq_len is parameterized for compatibility with the current project.
    """
    def __init__(self, num_inputs, seq_len, expanded_per_var=16):
        super(MyNet, self).__init__()
        self.num_inputs = num_inputs
        self.seq_len = seq_len
        self.expanded_per_var = expanded_per_var

        self.fc = nn.Linear(
            self.seq_len * self.num_inputs,
            self.seq_len * self.expanded_per_var * self.num_inputs
        )
        self.act = nn.ReLU()

    def forward(self, x):
        """
        x: [B, L, N]
        return: [B, L, expanded_per_var * N]
        """
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        x = self.act(x)
        x = x.contiguous().view(
            x.size(0),
            self.seq_len,
            self.expanded_per_var * self.num_inputs
        )
        return x


class VariableSelectionNetwork(nn.Module):
    """
    Paper-aligned selection structure adapted to preserve input/output shape.

    Input:
        embedding: [B, L, N]

    Internal:
        1) MyNet transforms input to [B, L, input_size * N]
        2) flattened_grn generates temporal selection scores over variables at each time step
        3) each variable branch uses its own GRN
        4) each variable branch is projected back to a scalar channel
        5) output shape is preserved as [B, L, N]
    """
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None, seq_len=96):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context = context
        self.seq_len = seq_len

        self.sparse_weights = torch.tensor(0.0)
        self.net = MyNet(
            num_inputs=self.num_inputs,
            seq_len=self.seq_len,
            expanded_per_var=self.input_size
        )

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(
                self.num_inputs * self.input_size,
                self.hidden_size,
                self.num_inputs,
                self.dropout,
                self.context,
                batch_first=True
            )
        else:
            self.flattened_grn = GatedResidualNetwork(
                self.num_inputs * self.input_size,
                self.hidden_size,
                self.num_inputs,
                self.dropout,
                batch_first=True
            )

        self.single_variable_grns = nn.ModuleList()
        self.single_variable_outs = nn.ModuleList()

        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(
                    self.input_size,
                    self.hidden_size,
                    self.hidden_size,
                    self.dropout,
                    batch_first=True
                )
            )
            self.single_variable_outs.append(
                TimeDistributed(
                    nn.Linear(self.hidden_size, 1),
                    batch_first=True
                )
            )

    def forward(self, embedding, context=None):
        """
        embedding: [B, L, N]

        Returns:
            outputs: [B, L, N]
            sparse_weights: [B, L, N]
            sparse_logits: [B, L, N]
        """
        # transformed inputs: [B, L, input_size * N]
        embedding = self.net(embedding)

        # selection logits branch
        if context is not None:
            sparse_logits = self.flattened_grn(embedding, context)   # [B, L, N]
        else:
            sparse_logits = self.flattened_grn(embedding)            # [B, L, N]

        sparse_weights = F.softmax(sparse_logits, dim=1)             # [B, L, N]

        # per-variable GRN branch
        var_outputs = []
        for i in range(self.num_inputs):
            var_slice = embedding[:, :, (i * self.input_size): (i + 1) * self.input_size]   # [B, L, input_size]
            var_hidden = self.single_variable_grns[i](var_slice)                              # [B, L, hidden_size]
            var_scalar = self.single_variable_outs[i](var_hidden)                             # [B, L, 1]
            var_outputs.append(var_scalar)

        # [B, L, N]
        var_outputs = torch.cat(var_outputs, dim=2)

        # preserve shape: [B, L, N]
        outputs = var_outputs * sparse_weights

        return outputs, sparse_weights, sparse_logits


class TemporalInputSelection(nn.Module):
    """
    Compatibility wrapper for the current project.

    Required current project output:
        {
            'selected_x': [B, L, N],
            'logits':     [B, N, L],
            'weights':    [B, N, L],
            'agg_repr':   [B, N],
            'sparse_loss': scalar
        }
    """

    def __init__(
        self,
        seq_len,
        hidden_dim=64,
        dropout=0.1,
        temperature=1.0,
        sparse_type='entropy',
        eps=1e-8
    ):
        super(TemporalInputSelection, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.temperature = temperature
        self.sparse_type = sparse_type
        self.eps = eps

        # aligned with provided code: internal per-variable expanded size
        self.input_size = 16

        # lazy build because num_inputs = N is only known at runtime
        self.vsn = None
        self.cached_num_inputs = None
        self.cached_seq_len = None

    def _build_if_needed(self, x):
        """
        x: [B, L, N]
        """
        _, seq_len, num_inputs = x.shape

        need_rebuild = (
            self.vsn is None
            or self.cached_num_inputs != num_inputs
            or self.cached_seq_len != seq_len
        )

        if need_rebuild:
            self.vsn = VariableSelectionNetwork(
                input_size=self.input_size,
                num_inputs=num_inputs,
                hidden_size=self.hidden_dim,
                dropout=self.dropout,
                context=None,
                seq_len=seq_len
            ).to(x.device)

            self.cached_num_inputs = num_inputs
            self.cached_seq_len = seq_len

    def _compute_sparse_loss(self, weights_bnL):
        """
        weights_bnL: [B, N, L]
        """
        if self.sparse_type == 'entropy':
            entropy = -(weights_bnL * torch.log(weights_bnL + self.eps)).sum(dim=-1)  # [B, N]
            sparse_loss = entropy.mean()
        elif self.sparse_type == 'logits_l1':
            sparse_loss = torch.mean(torch.abs(weights_bnL))
        elif self.sparse_type == 'none':
            sparse_loss = weights_bnL.new_tensor(0.0)
        else:
            raise ValueError(f"Unsupported sparse_type: {self.sparse_type}")
        return sparse_loss

    def forward(self, x):
        """
        x: [B, L, N]

        Returns:
            {
                'selected_x': [B, L, N],
                'logits':     [B, N, L],
                'weights':    [B, N, L],
                'agg_repr':   [B, N],
                'sparse_loss': scalar
            }
        """
        self._build_if_needed(x)

        selected_x, sparse_weights, sparse_logits = self.vsn(x)
        # selected_x:    [B, L, N]
        # sparse_weights:[B, L, N]
        # sparse_logits: [B, L, N]

        # project to current project convention
        weights = sparse_weights.permute(0, 2, 1)  # [B, N, L]
        logits = sparse_logits.permute(0, 2, 1)    # [B, N, L]

        # compatibility summary statistic
        agg_repr = torch.sum(weights * x.permute(0, 2, 1), dim=-1)  # [B, N]

        sparse_loss = self._compute_sparse_loss(weights)

        return {
            'selected_x': selected_x,   # [B, L, N]
            'logits': logits,           # [B, N, L]
            'weights': weights,         # [B, N, L]
            'agg_repr': agg_repr,       # [B, N]
            'sparse_loss': sparse_loss
        }
