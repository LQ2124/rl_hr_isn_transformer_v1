import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.temporal_selection import TemporalInputSelection


class Model(nn.Module):
    """
    Static temporal selection + iTransformer backbone.

    Forecasting path:
        x_enc
        -> normalization
        -> static temporal input selection
        -> DataEmbedding_inverted
        -> Encoder
        -> projection
        -> de-normalization
        -> pred, aux
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Optional selector configs with safe fallbacks
        selector_hidden_dim = getattr(configs, 'selector_hidden_dim', 64)
        selector_dropout = getattr(configs, 'selector_dropout', configs.dropout)
        selector_temperature = getattr(configs, 'selector_temperature', 1.0)
        sparse_type = getattr(configs, 'sparse_type', 'entropy')

        # Static temporal input selection
        self.temporal_selector = TemporalInputSelection(
            seq_len=configs.seq_len,
            hidden_dim=selector_hidden_dim,
            dropout=selector_dropout,
            temperature=selector_temperature,
            sparse_type=sparse_type
        )

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Encoder
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

        # Decoder / task heads
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
        means: [B, 1, N]
        stdev: [B, 1, N]
        """
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, out_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, out_len, 1))
        return x

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [B, L, N]
        """
        # Normalization from Non-stationary Transformer
        x_enc, means, stdev = self._normalize(x_enc)
        _, _, N = x_enc.shape

        # Static temporal input selection
        sel_out = self.temporal_selector(x_enc)
        selected_x = sel_out['selected_x']  # [B, L, N]

        # Original iTransformer backbone
        enc_out = self.enc_embedding(selected_x, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-normalization
        dec_out = self._denormalize(dec_out, means, stdev, self.pred_len)

        aux = {
            'selection_logits': sel_out['logits'],
            'selection_weights': sel_out['weights'],
            'agg_repr': sel_out['agg_repr'],
            'sparse_loss': sel_out['sparse_loss'],
            'selected_x': selected_x
        }
        return dec_out, aux

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # For the static reconstruction stage, only forecasting is the primary target.
        # Keep behavior close to original implementation for compatibility.
        x_enc, means, stdev = self._normalize(x_enc)
        _, L, N = x_enc.shape

        sel_out = self.temporal_selector(x_enc)
        selected_x = sel_out['selected_x']

        enc_out = self.enc_embedding(selected_x, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denormalize(dec_out, means, stdev, L)

        aux = {
            'selection_logits': sel_out['logits'],
            'selection_weights': sel_out['weights'],
            'agg_repr': sel_out['agg_repr'],
            'sparse_loss': sel_out['sparse_loss'],
            'selected_x': selected_x
        }
        return dec_out, aux

    def anomaly_detection(self, x_enc):
        x_enc, means, stdev = self._normalize(x_enc)
        _, L, N = x_enc.shape

        sel_out = self.temporal_selector(x_enc)
        selected_x = sel_out['selected_x']

        enc_out = self.enc_embedding(selected_x, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self._denormalize(dec_out, means, stdev, L)

        aux = {
            'selection_logits': sel_out['logits'],
            'selection_weights': sel_out['weights'],
            'agg_repr': sel_out['agg_repr'],
            'sparse_loss': sel_out['sparse_loss'],
            'selected_x': selected_x
        }
        return dec_out, aux

    def classification(self, x_enc, x_mark_enc):
        sel_out = self.temporal_selector(x_enc)
        selected_x = sel_out['selected_x']

        enc_out = self.enc_embedding(selected_x, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        aux = {
            'selection_logits': sel_out['logits'],
            'selection_weights': sel_out['weights'],
            'agg_repr': sel_out['agg_repr'],
            'sparse_loss': sel_out['sparse_loss'],
            'selected_x': selected_x
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
