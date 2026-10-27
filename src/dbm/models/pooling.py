"""Temporal pooling layers for sequence classifiers."""

from __future__ import annotations

import torch
from torch import nn


class TemporalPool(nn.Module):
    """Pool sequence outputs using last, mean, or learned attention."""

    def __init__(self, mode: str, hidden_size: int):
        super().__init__()
        if mode not in {"last", "mean", "attention"}:
            raise ValueError(f"Unknown pooling mode: {mode}")
        self.mode = mode
        self.attention = (nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
                          if mode == "attention" else None)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor | None = None):
        """Return pooled vectors and optional attention weights."""
        batch, steps, _ = sequence.shape
        mask = torch.arange(steps, device=sequence.device)[None, :] < lengths[:, None] if lengths is not None else None
        if self.mode == "last":
            indices = lengths.to(sequence.device) - 1 if lengths is not None else torch.full((batch,), steps - 1, device=sequence.device)
            return sequence[torch.arange(batch, device=sequence.device), indices], None
        if self.mode == "mean":
            if mask is None:
                return sequence.mean(dim=1), None
            valid = mask.unsqueeze(-1)
            return (sequence * valid).sum(dim=1) / lengths.to(sequence.device).clamp_min(1).unsqueeze(-1), None
        scores = self.attention(sequence).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=1)
        return torch.sum(sequence * alpha.unsqueeze(-1), dim=1), alpha

