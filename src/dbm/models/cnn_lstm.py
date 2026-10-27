"""CNN-LSTM classifier with a shared frame encoder."""

from __future__ import annotations

import torch
from torch import nn

from dbm.labels import num_classes
from dbm.models.cnn_backbone import CNNBackbone
from dbm.models.pooling import TemporalPool


class CNNLSTM(nn.Module):
    """Classify (batch, time, channels, height, width) clips."""

    def __init__(self, channels: tuple[int, ...] | list[int] = (32, 64, 128, 256),
                 input_channels: int = 3, hidden_size: int = 128, num_layers: int = 1,
                 lstm_dropout: float = 0.0, head_dropout: float = 0.4,
                 pooling: str = "mean", n_classes: int | None = None):
        super().__init__()
        self.backbone = CNNBackbone(channels, input_channels)
        self.lstm = nn.LSTM(self.backbone.output_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout if num_layers > 1 else 0.0)
        self.pool = TemporalPool(pooling, hidden_size)
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden_size, n_classes or num_classes())

    def forward(self, clips: torch.Tensor, return_attention: bool = False):
        """Return logits for each clip, optionally returning temporal weights."""
        batch, steps, channels, height, width = clips.shape
        encoded = self.backbone(clips.reshape(batch * steps, channels, height, width))
        sequence, _ = self.lstm(encoded.reshape(batch, steps, -1))
        pooled, alpha = self.pool(sequence)
        logits = self.classifier(self.dropout(pooled))
        return (logits, alpha) if return_attention else logits
