"""Relation-aware graph network for admission-level medication prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from medrec.models.common import AdaptiveModalityFusion


class RelationAwareLayer(nn.Module):
    """Aggregate separately transformed messages from named edge relations."""

    def __init__(self, hidden_dim: int, relation_names: Sequence[str]) -> None:
        super().__init__()
        if not relation_names:
            raise ValueError("at least one relation is required")
        self.relation_names = tuple(relation_names)
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.relation_projections = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in self.relation_names}
        )
        self.relation_logits = nn.Parameter(torch.zeros(len(self.relation_names)))

    def forward(
        self,
        values: torch.Tensor,
        relation_edges: Mapping[str, torch.Tensor],
        relation_weights: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        output = self.self_projection(values)
        relation_scale = torch.softmax(self.relation_logits, dim=0) * len(self.relation_names)

        for index, name in enumerate(self.relation_names):
            if name not in relation_edges:
                continue
            edge_index = relation_edges[name]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"{name} edge_index must have shape [2, edges]")
            source, destination = edge_index
            messages = self.relation_projections[name](values[source])
            if relation_weights is not None and name in relation_weights:
                weights = relation_weights[name].reshape(-1, 1).to(messages.dtype)
                if weights.shape[0] != messages.shape[0]:
                    raise ValueError(f"{name} edge weights do not match its edge count")
                messages = messages * weights

            aggregate = torch.zeros_like(values)
            aggregate.index_add_(0, destination, messages)
            degree = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=values.dtype))
            aggregate = aggregate / degree.clamp_min(1).unsqueeze(1)
            output = output + relation_scale[index] * aggregate
        return output


class MedicationRGCN(nn.Module):
    """Multi-modal relation-aware GCN with residual message passing."""

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        relation_names: Sequence[str],
        num_labels: int,
        *,
        hidden_dim: int = 256,
        projection_dim: int = 64,
        num_layers: int = 2,
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
        self.relation_layers = nn.ModuleList(
            [RelationAwareLayer(hidden_dim, relation_names) for _ in range(num_layers)]
        )
        self.normalizations = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        relation_edges: Mapping[str, torch.Tensor],
        relation_weights: Mapping[str, torch.Tensor] | None = None,
        availability: torch.Tensor | None = None,
        *,
        return_gates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden, gates = self.fusion(features, availability)
        for layer, normalization in zip(self.relation_layers, self.normalizations):
            update = layer(hidden, relation_edges, relation_weights)
            hidden = normalization(hidden + self.dropout(torch.nn.functional.gelu(update)))
        logits = self.classifier(hidden)
        return (logits, gates) if return_gates else logits
