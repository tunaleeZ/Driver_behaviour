"""Unidirectional LSTM classifier for geometric and landmark sequences."""

from __future__ import annotations

import torch
from torch import nn

from dbm.labels import num_classes
from dbm.models.pooling import TemporalPool


class LSTMClassifier(nn.Module):
    """Classify (batch, time, features) sequences and return logits."""

    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.3, pooling: str = "mean", n_classes: int | None = None):
        super().__init__()
        if input_dim <= 0 or num_layers <= 0:
            raise ValueError("input_dim and num_layers must be positive")
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
        self.pool = TemporalPool(pooling, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes or num_classes())

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None, return_attention: bool = False):
        """Return logits, optionally with temporal attention weights."""
        sequence, _ = self.lstm(x)
        pooled, alpha = self.pool(sequence, lengths)
        logits = self.classifier(self.dropout(pooled))
        return (logits, alpha) if return_attention else logits

