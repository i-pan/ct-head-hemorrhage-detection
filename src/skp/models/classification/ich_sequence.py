"""Contextual slice and series classifier for frozen RSNA ICH features."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ClassAttentionPool(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int, num_classes: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, num_classes),
        )
        self.classifier_weight = nn.Parameter(
            torch.empty(num_classes, feature_dim)
        )
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.classifier_weight)

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(features)
        scores = scores.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf)
        attention = scores.softmax(dim=1)
        pooled = torch.einsum("btc,bth->bch", attention, features)
        logits = torch.einsum(
            "bch,ch->bc", pooled, self.classifier_weight
        ) + self.classifier_bias
        return logits, attention


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg.num_classes)
        self.input_norm = nn.LayerNorm(cfg.feature_dim)
        self.projection = nn.Linear(cfg.feature_dim, cfg.sequence_projection_dim)
        self.projection_norm = nn.LayerNorm(cfg.sequence_projection_dim)
        self.projection_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(cfg.feature_dropout)
        self.feature_noise_std = float(cfg.feature_noise_std)
        self.slice_dropout_p = float(cfg.slice_feature_dropout)
        self.sequence_architecture = cfg.get("sequence_architecture", "gru").lower()
        if self.sequence_architecture in {"gru", "lstm"}:
            recurrent_cls = nn.GRU if self.sequence_architecture == "gru" else nn.LSTM
            self.sequence = recurrent_cls(
                input_size=cfg.sequence_projection_dim + 1,
                hidden_size=cfg.sequence_hidden_dim,
                num_layers=cfg.sequence_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=cfg.sequence_dropout if cfg.sequence_num_layers > 1 else 0.0,
            )
            self.position_projection = None
            contextual_dim = cfg.sequence_hidden_dim * 2
        elif self.sequence_architecture == "transformer":
            self.position_projection = nn.Linear(1, cfg.sequence_projection_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.sequence_projection_dim,
                nhead=cfg.transformer_num_heads,
                dim_feedforward=cfg.transformer_feedforward_dim,
                dropout=cfg.sequence_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.sequence_num_layers,
                norm=nn.LayerNorm(cfg.sequence_projection_dim),
                enable_nested_tensor=False,
            )
            contextual_dim = cfg.sequence_projection_dim
        else:
            raise ValueError(
                "sequence_architecture must be one of: gru, lstm, transformer; "
                f"got {self.sequence_architecture!r}"
            )
        self.head_dropout = nn.Dropout(cfg.dropout)
        self.slice_head = nn.Linear(contextual_dim, self.num_classes)
        self.sequence_pool = ClassAttentionPool(
            contextual_dim, cfg.attention_dim, self.num_classes
        )
        self.mil_pool = ClassAttentionPool(
            cfg.sequence_projection_dim, cfg.attention_dim, self.num_classes
        )
        self.criterion = None

    def _augment_features(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if not self.training:
            return features
        if self.feature_noise_std > 0:
            noise = torch.randn_like(features) * self.feature_noise_std
            features = features + noise * valid_mask.unsqueeze(-1)
        if self.slice_dropout_p > 0:
            keep = torch.rand(
                features.shape[:2], device=features.device
            ) >= self.slice_dropout_p
            features = features * keep.unsqueeze(-1)
        return self.feature_dropout(features)

    def forward(
        self, batch: Dict, return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        valid_mask = batch["valid_mask"].bool()
        lengths = batch["length"].long()
        features = self.input_norm(batch["x"].float())
        features = self.projection_activation(self.projection(features))
        features = self.projection_norm(features)
        features = self._augment_features(features, valid_mask)

        position = batch["position"].float().unsqueeze(-1)
        if self.sequence_architecture == "transformer":
            sequence_input = features + self.position_projection(position)
            contextual = self.sequence(
                sequence_input, src_key_padding_mask=~valid_mask
            )
        else:
            sequence_input = torch.cat([features, position], dim=-1)
            packed = pack_padded_sequence(
                sequence_input,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.sequence(packed)
            contextual, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=sequence_input.shape[1],
            )
        slice_logits = self.slice_head(self.head_dropout(contextual))
        series_logits, sequence_attention = self.sequence_pool(
            contextual, valid_mask
        )
        mil_logits, mil_attention = self.mil_pool(features, valid_mask)

        out = {
            "slice_logits": slice_logits,
            "series_logits": series_logits,
            "mil_logits": mil_logits,
            "sequence_attention": sequence_attention,
            "mil_attention": mil_attention,
        }
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
