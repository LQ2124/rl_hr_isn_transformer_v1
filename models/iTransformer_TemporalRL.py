import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.temporal_selection import TemporalInputSelection
from layers.dynamic_selection_rl import DynamicSelectionController


class Model(nn.Module):
    """
    Dynamic temporal selection + lightweight actor-critic + iTransformer backbone

    Upgraded design:
        RL controls target-aware multi-scale scale preference,
        instead of weak point-wise affine logit perturbation.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # -------- Static selector config --------
        selector_hidden_dim = getattr(configs, 'selector_hidden_dim', 64)
        selector_dropout = getattr(configs, 'selector_dropout', configs.dropout)
        selector_temperature = getattr(configs, 'selector_temperature', 1.0)
        sparse_type = getattr(configs, 'sparse_type', 'entropy')
        residual_scale = getattr(configs, 'residual_scale', 0.5)

        local_kernel_size = getattr(configs, 'local_kernel_size', 5)
        coarse_ratio = getattr(configs, 'coarse_ratio', 4)

        # -------- Dynamic controller config --------
        actor_hidden_dim = getattr(configs, 'actor_hidden_dim', 64)
        critic_hidden_dim = getattr(configs, 'critic_hidden_dim', 64)
        action_scale = getattr(configs, 'action_scale', 0.5)

        # -------- Static temporal selector --------
        self.temporal_selector = TemporalInputSelection(
            seq_len=configs.seq_len,
            hidden_dim=selector_hidden_dim,
            dropout=selector_dropout,
            temperature=selector_temperature,
            sparse_type=sparse_type,
            residual_scale=residual_scale,
            local_kernel_size=local_kernel_size,
            coarse_ratio=coarse_ratio
        )

        # -------- Dynamic RL controller --------
        self.dynamic_controller = DynamicSelectionController(
            seq_len=configs.seq_len,
            actor_hidden_dim=actor_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
            action_scale=action_scale,
            temperature=selector_temperature,
            residual_scale=residual_scale
        )

        # -------- iTransformer backbone --------
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # -------- Projection head --------
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def _normalize(self, x):
        """
        x: [B, L, N]
        """
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        return x, means, stdev

    def _denormalize(self, x, means, stdev, out_len):
        """
        x: [B, out_len, N]
        """
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, out_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, out_len, 1))
        return x

    def _backbone_predict(self, selected_x, x_mark_enc, means, stdev, out_len):
        """
        selected_x: [B, L, N]
        returns:
            pred: [B, out_len, N]
        """
        _, _, N = selected_x.shape

        enc_out = self.enc_embedding(selected_x, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        pred = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        pred = self._denormalize(pred, means, stdev, out_len)
        return pred

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Dynamic forecasting path.
        Returns:
            dynamic_pred: [B, pred_len, N]
            aux: dict containing static/dynamic intermediate quantities
        """
        # -------- Normalize input --------
        x_norm, means, stdev = self._normalize(x_enc)

        # -------- Static multi-scale selector --------
        static_out = self.temporal_selector(x_norm)

        static_selected_x = static_out['selected_x']          # [B, L, N]
        base_logits = static_out['logits']                    # [B, N, L]
        base_weights = static_out['weights']                  # [B, N, L]
        agg_repr = static_out['agg_repr']                     # [B, N]
        sparse_loss = static_out['sparse_loss']

        branch_logits = static_out['branch_logits']           # dict of [B, N, L]
        static_scale_weights = static_out['scale_weights']    # [B, N, 3]
        summaries = static_out['summaries']                   # dict of [B, N]

        # -------- Static prediction branch --------
        static_pred = self._backbone_predict(
            static_selected_x,
            x_mark_enc,
            means,
            stdev,
            self.pred_len
        )  # [B, pred_len, N]

        # -------- Dynamic RL controller --------
        dynamic_out = self.dynamic_controller(
            x=x_norm,
            branch_logits=branch_logits,
            base_weights=base_weights,
            agg_repr=agg_repr,
            static_scale_weights=static_scale_weights,
            summaries=summaries
        )

        dynamic_selected_x = dynamic_out['selected_x']                # [B, L, N]
        state = dynamic_out['state']                                  # [B, N, state_dim]
        value = dynamic_out['value']                                  # [B, N, 1]

        action_delta = dynamic_out['action_delta']                    # [B, N, 3]
        action_strength = dynamic_out['action_strength']              # [B, N]

        dynamic_scale_logits = dynamic_out['dynamic_scale_logits']    # [B, N, 3]
        dynamic_scale_weights = dynamic_out['dynamic_scale_weights']  # [B, N, 3]

        calibrated_logits = dynamic_out['calibrated_logits']          # [B, N, L]
        dynamic_weights = dynamic_out['dynamic_weights']              # [B, N, L]

        # -------- Dynamic prediction branch --------
        dynamic_pred = self._backbone_predict(
            dynamic_selected_x,
            x_mark_enc,
            means,
            stdev,
            self.pred_len
        )  # [B, pred_len, N]

        aux = {
            # static outputs
            'selection_logits': base_logits,
            'selection_weights': base_weights,
            'agg_repr': agg_repr,
            'sparse_loss': sparse_loss,
            'static_selected_x': static_selected_x,

            # static multi-scale info
            'branch_logits': branch_logits,
            'static_scale_weights': static_scale_weights,
            'summaries': summaries,

            # dynamic RL outputs
            'state': state,
            'value': value,
            'action_delta': action_delta,
            'action_strength': action_strength,
            'dynamic_scale_logits': dynamic_scale_logits,
            'dynamic_scale_weights': dynamic_scale_weights,
            'calibrated_logits': calibrated_logits,
            'dynamic_weights': dynamic_weights,
            'selected_x': dynamic_selected_x,

            # explicit predictions for trainer-side reward
            'static_pred': static_pred,
            'dynamic_pred': dynamic_pred
        }

        return dynamic_pred, aux

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_norm, means, stdev = self._normalize(x_enc)
        _, L, N = x_norm.shape

        static_out = self.temporal_selector(x_norm)

        dynamic_out = self.dynamic_controller(
            x=x_norm,
            branch_logits=static_out['branch_logits'],
            base_weights=static_out['weights'],
            agg_repr=static_out['agg_repr'],
            static_scale_weights=static_out['scale_weights'],
            summaries=static_out['summaries']
        )

        dynamic_selected_x = dynamic_out['selected_x']
        pred = self._backbone_predict(dynamic_selected_x, x_mark_enc, means, stdev, L)

        aux = {
            'selection_logits': static_out['logits'],
            'selection_weights': static_out['weights'],
            'agg_repr': static_out['agg_repr'],
            'sparse_loss': static_out['sparse_loss'],
            'branch_logits': static_out['branch_logits'],
            'static_scale_weights': static_out['scale_weights'],
            'summaries': static_out['summaries'],

            'state': dynamic_out['state'],
            'value': dynamic_out['value'],
            'action_delta': dynamic_out['action_delta'],
            'action_strength': dynamic_out['action_strength'],
            'dynamic_scale_logits': dynamic_out['dynamic_scale_logits'],
            'dynamic_scale_weights': dynamic_out['dynamic_scale_weights'],
            'calibrated_logits': dynamic_out['calibrated_logits'],
            'dynamic_weights': dynamic_out['dynamic_weights'],
            'selected_x': dynamic_selected_x,

            'static_pred': pred.detach(),
            'dynamic_pred': pred
        }
        return pred, aux

    def anomaly_detection(self, x_enc):
        x_norm, means, stdev = self._normalize(x_enc)
        _, L, N = x_norm.shape

        static_out = self.temporal_selector(x_norm)

        dynamic_out = self.dynamic_controller(
            x=x_norm,
            branch_logits=static_out['branch_logits'],
            base_weights=static_out['weights'],
            agg_repr=static_out['agg_repr'],
            static_scale_weights=static_out['scale_weights'],
            summaries=static_out['summaries']
        )

        dynamic_selected_x = dynamic_out['selected_x']
        pred = self._backbone_predict(dynamic_selected_x, None, means, stdev, L)

        aux = {
            'selection_logits': static_out['logits'],
            'selection_weights': static_out['weights'],
            'agg_repr': static_out['agg_repr'],
            'sparse_loss': static_out['sparse_loss'],
            'branch_logits': static_out['branch_logits'],
            'static_scale_weights': static_out['scale_weights'],
            'summaries': static_out['summaries'],

            'state': dynamic_out['state'],
            'value': dynamic_out['value'],
            'action_delta': dynamic_out['action_delta'],
            'action_strength': dynamic_out['action_strength'],
            'dynamic_scale_logits': dynamic_out['dynamic_scale_logits'],
            'dynamic_scale_weights': dynamic_out['dynamic_scale_weights'],
            'calibrated_logits': dynamic_out['calibrated_logits'],
            'dynamic_weights': dynamic_out['dynamic_weights'],
            'selected_x': dynamic_selected_x,

            'static_pred': pred.detach(),
            'dynamic_pred': pred
        }
        return pred, aux

    def classification(self, x_enc, x_mark_enc):
        static_out = self.temporal_selector(x_enc)

        dynamic_out = self.dynamic_controller(
            x=x_enc,
            branch_logits=static_out['branch_logits'],
            base_weights=static_out['weights'],
            agg_repr=static_out['agg_repr'],
            static_scale_weights=static_out['scale_weights'],
            summaries=static_out['summaries']
        )

        dynamic_selected_x = dynamic_out['selected_x']

        enc_out = self.enc_embedding(dynamic_selected_x, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        aux = {
            'selection_logits': static_out['logits'],
            'selection_weights': static_out['weights'],
            'agg_repr': static_out['agg_repr'],
            'sparse_loss': static_out['sparse_loss'],
            'branch_logits': static_out['branch_logits'],
            'static_scale_weights': static_out['scale_weights'],
            'summaries': static_out['summaries'],

            'state': dynamic_out['state'],
            'value': dynamic_out['value'],
            'action_delta': dynamic_out['action_delta'],
            'action_strength': dynamic_out['action_strength'],
            'dynamic_scale_logits': dynamic_out['dynamic_scale_logits'],
            'dynamic_scale_weights': dynamic_out['dynamic_scale_weights'],
            'calibrated_logits': dynamic_out['calibrated_logits'],
            'dynamic_weights': dynamic_out['dynamic_weights'],
            'selected_x': dynamic_selected_x,

            'static_pred': output.detach(),
            'dynamic_pred': output
        }
        return output, aux

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, aux = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], aux
        if self.task_name == 'imputation':
            dec_out, aux = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out, aux
        if self.task_name == 'anomaly_detection':
            dec_out, aux = self.anomaly_detection(x_enc)
            return dec_out, aux
        if self.task_name == 'classification':
            dec_out, aux = self.classification(x_enc, x_mark_enc)
            return dec_out, aux
        return None
