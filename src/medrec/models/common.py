"""Shared neural-network components for multi-modal admission features."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ModalityEncoder(nn.Module):
    """Project one feature group into a common latent dimension."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class AdaptiveModalityFusion(nn.Module):
    """Encode feature groups and learn an admission-specific weight for each group.

    Gate weights are normalized with a softmax and scaled so that neutral gates
    have a value of one. This keeps the initial feature scale comparable to an
    ungated concatenation.
    """

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        *,
        projection_dim: int = 64,
        output_dim: int = 256,
        gate_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("at least one modality is required")

        self.names = tuple(modality_dims)
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(width, projection_dim, dropout)
                for name, width in modality_dims.items()
            }
        )
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(projection_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for name in self.names
            }
        )
        for gate in self.gates.values():
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

        self.merge = nn.Sequential(
            nn.Linear(len(self.names) * projection_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        availability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [name for name in self.names if name not in features]
        if missing:
            raise ValueError(f"missing modality tensors: {missing}")

        encoded = [self.encoders[name](features[name]) for name in self.names]
        batch_size = encoded[0].shape[0]
        if any(value.shape[0] != batch_size for value in encoded):
            raise ValueError("all modality tensors must have the same row count")

        gate_logits = torch.cat(
            [self.gates[name](value) for name, value in zip(self.names, encoded)], dim=1
        )
        if availability is None:
            availability = torch.ones_like(gate_logits, dtype=torch.bool)
        else:
            availability = availability.to(device=gate_logits.device, dtype=torch.bool)
            if availability.shape != gate_logits.shape:
                raise ValueError("availability must have shape [rows, modalities]")
            if (~availability).all(dim=1).any():
                raise ValueError("every row must contain at least one available modality")

        masked_logits = gate_logits.masked_fill(~availability, torch.finfo(gate_logits.dtype).min)
        active_count = availability.sum(dim=1, keepdim=True).to(gate_logits.dtype)
        gate_values = torch.softmax(masked_logits, dim=1) * active_count
        weighted = [value * gate_values[:, index : index + 1] for index, value in enumerate(encoded)]
        return self.merge(torch.cat(weighted, dim=1)), gate_values


def split_modalities(
    values: torch.Tensor,
    feature_slices: Mapping[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Split a dense visit matrix into named, non-overlapping feature groups."""
    result: dict[str, torch.Tensor] = {}
    previous_end = 0
    for name, (start, end) in feature_slices.items():
        if start != previous_end or end <= start:
            raise ValueError("feature slices must be ordered, contiguous, and non-empty")
        result[name] = values[:, start:end]
        previous_end = end
    if previous_end != values.shape[1]:
        raise ValueError("feature slices must cover the complete feature matrix")
    return result
