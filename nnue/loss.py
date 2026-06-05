# nnue/loss.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueLoss(nn.Module):
    """
    Standard NNUE value loss.

    Mean Squared Error between predicted evaluation and target value.
    Supports sample weights to emphasize deep search or high-ply game phases.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Ensure dimensions match for element-wise operations
        loss = (predictions.view_as(targets) - targets) ** 2

        if weights is not None:
            loss = loss * weights.view_as(loss)

        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class HuberValueLoss(nn.Module):
    """
    Robust alternative to MSE.
    Less sensitive to noisy outliers or engine search blunder evaluations.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:

        loss = F.huber_loss(
            predictions.view_as(targets),
            targets,
            reduction="none",
            delta=self.delta,
        )

        if weights is not None:
            loss = loss * weights.view_as(loss)

        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class WinDrawLoss(nn.Module):
    """
    Optional multi-class classification loss.

    Classes must map strictly to non-negative target indices for PyTorch:
         0 -> Loss
         1 -> Draw
         2 -> Win
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits: Raw, unnormalized network scores shape [Batch, 3]
        labels: Class index long tensor shape [Batch] matching [0, 1, 2]
        """
        return self.loss_fn(logits, labels.long())


class CombinedLoss(nn.Module):
    """
    Multi-objective loss function combining MSE and Huber loss stability.
    """

    def __init__(self, value_weight: float = 1.0, huber_weight: float = 0.0) -> None:
        super().__init__()
        self.value_weight = value_weight
        self.huber_weight = huber_weight

        self.value_loss = ValueLoss(reduction="none")
        self.huber_loss = HuberValueLoss(reduction="none")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:

        total_loss = torch.zeros_like(predictions)

        if self.value_weight > 0:
            total_loss += self.value_weight * self.value_loss(predictions, targets, weights)

        if self.huber_weight > 0:
            total_loss += self.huber_weight * self.huber_loss(predictions, targets, weights)

        return total_loss.mean()
