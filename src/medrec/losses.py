"""Loss functions for accurate and safety-aware medication prediction."""

from __future__ import annotations

import torch
from torch import nn


class ExpectedDDIRisk(nn.Module):
    """Differentiable expected interaction rate under predicted probabilities.

    The adjacency matrix must be symmetric, binary, and aligned with the model's
    medication-label order. The loss divides expected interacting pairs by all
    expected medication pairs for each admission before averaging the batch.
    """

    def __init__(self, adjacency: torch.Tensor, epsilon: float = 1e-8) -> None:
        super().__init__()
        matrix = torch.as_tensor(adjacency, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if not torch.equal(matrix, matrix.T):
            raise ValueError("adjacency must be symmetric")
        if not torch.allclose(torch.diagonal(matrix), torch.zeros(matrix.shape[0])):
            raise ValueError("adjacency diagonal must be zero")
        if not torch.logical_or(matrix == 0, matrix == 1).all():
            raise ValueError("adjacency must be binary")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.register_buffer("adjacency", matrix)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.adjacency.shape[0]:
            raise ValueError("logits and DDI adjacency dimensions do not match")
        probabilities = torch.sigmoid(logits.float())
        interacting_pairs = 0.5 * torch.einsum(
            "bi,ij,bj->b", probabilities, self.adjacency, probabilities
        )
        probability_sum = probabilities.sum(dim=1)
        all_pairs = 0.5 * (
            probability_sum.square() - probabilities.square().sum(dim=1)
        )
        return (interacting_pairs / (all_pairs + self.epsilon)).mean()


class SafetyAwareBCELoss(nn.Module):
    """Binary cross-entropy with an optional expected DDI-risk penalty."""

    def __init__(
        self,
        ddi_adjacency: torch.Tensor | None = None,
        *,
        ddi_weight: float = 0.0,
        positive_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if ddi_weight < 0:
            raise ValueError("ddi_weight must be non-negative")
        if ddi_weight > 0 and ddi_adjacency is None:
            raise ValueError("ddi_adjacency is required when ddi_weight is positive")
        self.bce = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        self.ddi = ExpectedDDIRisk(ddi_adjacency) if ddi_adjacency is not None else None
        self.ddi_weight = float(ddi_weight)

    def components(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have identical shapes")
        bce_loss = self.bce(logits.float(), targets.float())
        ddi_loss = logits.new_zeros(()) if self.ddi is None else self.ddi(logits)
        total_loss = bce_loss + self.ddi_weight * ddi_loss
        return total_loss, bce_loss, ddi_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.components(logits, targets)[0]
