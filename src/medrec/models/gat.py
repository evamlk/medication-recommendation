"""Graph attention network for admission-level medication prediction."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from medrec.models.common import AdaptiveModalityFusion


class MedicationGAT(nn.Module):
    """Multi-modal GATv2 with residual attention blocks."""

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        num_labels: int,
        *,
        hidden_dim: int = 256,
        projection_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1 or num_labels < 1 or heads < 1:
            raise ValueError("num_layers, num_labels, and heads must be positive")
        self.fusion = AdaptiveModalityFusion(
            modality_dims,
            projection_dim=projection_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        self.attention_layers = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=1,
                    add_self_loops=True,
                    fill_value=1.0,
                )
                for _ in range(num_layers)
            ]
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
        edge_attr = None if edge_weight is None else edge_weight.reshape(-1, 1)
        for attention, normalization in zip(self.attention_layers, self.normalizations):
            update = attention(hidden, edge_index, edge_attr=edge_attr)
            hidden = normalization(hidden + self.dropout(torch.nn.functional.gelu(update)))
        logits = self.classifier(hidden)
        return (logits, gates) if return_gates else logits
