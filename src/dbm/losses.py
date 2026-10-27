"""Classification losses. Cost matrix rows are true labels, columns predictions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Focal loss; noisy labels may receive disproportionate emphasis."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class CostSensitiveLoss(nn.Module):
    """Base loss plus expected configured cost; costs require group justification."""

    def __init__(self, cost_matrix: torch.Tensor, base_loss: nn.Module | None = None, weight: float = 0.5):
        super().__init__()
        self.register_buffer("cost_matrix", cost_matrix.float())
        self.base_loss = base_loss or nn.CrossEntropyLoss()
        self.weight = float(weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1)
        expected_cost = self.cost_matrix[targets] * probs
        return self.base_loss(logits, targets) + self.weight * expected_cost.sum(dim=-1).mean()


def build_loss(name: str, class_weights: torch.Tensor | None = None, gamma: float = 2.0,
               cost_matrix: torch.Tensor | None = None, cost_lambda: float = 0.5) -> nn.Module:
    """Create a configured loss module."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=class_weights)
    if name == "cost_sensitive":
        if cost_matrix is None:
            raise ValueError("cost_matrix is required for cost_sensitive loss")
        return CostSensitiveLoss(cost_matrix, nn.CrossEntropyLoss(weight=class_weights), cost_lambda)
    raise ValueError(f"Unknown loss: {name}")


def class_weights_from_labels(labels: list[int], n_classes: int, cap: float | None = None) -> torch.Tensor:
    """Compute normalized inverse-frequency weights from training labels only."""
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=n_classes).float()
    weights = counts.sum() / (n_classes * counts.clamp_min(1.0))
    weights[counts == 0] = 0.0
    if cap is not None:
        weights = weights.clamp(max=float(cap))
    return weights

