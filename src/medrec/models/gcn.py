"""Graph convolutional network for admission-level medication prediction."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from medrec.models.common import AdaptiveModalityFusion


class MedicationGCN(nn.Module):
    """Multi-modal GCN with residual graph-convolution blocks."""

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        num_labels: int,
        *,
        hidden_dim: int = 256,
        projection_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1 or num_labels < 1:
            raise ValueError("num_layers and num_labels must be positive")
        self.fusion = AdaptiveModalityFusion(
            modality_dims,
            projection_dim=projection_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.convolutions = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True) for _ in range(num_layers)]
        )
        self.normalizations = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        availability: torch.Tensor | None = None,
        *,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden, gates = self.fusion(features, availability)
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            update = convolution(hidden, edge_index, edge_weight=edge_weight)
            hidden = normalization(hidden + self.dropout(torch.nn.functional.gelu(update)))
        logits = self.classifier(hidden)
        return (logits, gates) if return_gates else logits
