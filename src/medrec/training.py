"""Small, reusable helpers for reproducible model training."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class EarlyStopping:
    """Track a maximized validation score and retain the best model state."""

    patience: int = 10
    minimum_improvement: float = 0.0
    best_score: float = field(default=-float("inf"), init=False)
    bad_epochs: int = field(default=0, init=False)
    best_state: dict[str, torch.Tensor] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least one")
        if self.minimum_improvement < 0:
            raise ValueError("minimum_improvement must be non-negative")

    def update(self, score: float, model: nn.Module) -> bool:
        """Record a score and return true when training should stop."""
        if not np.isfinite(score):
            raise ValueError("validation score must be finite")
        if score > self.best_score + self.minimum_improvement:
            self.best_score = float(score)
            self.bad_epochs = 0
            self.best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is None:
            raise RuntimeError("no best state has been recorded")
        model.load_state_dict(copy.deepcopy(self.best_state))


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    targets: torch.Tensor,
    *,
    model_args: Sequence[Any] = (),
    model_kwargs: Mapping[str, Any] | None = None,
    gradient_clip_norm: float | None = 5.0,
) -> float:
    """Run one optimization step and return its scalar loss."""
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(*model_args, **dict(model_kwargs or {}))
    logits = output[0] if isinstance(output, tuple) else output
    loss = loss_function(logits, targets)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("loss must be a finite scalar")
    loss.backward()
    if gradient_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    *,
    model_args: Sequence[Any] = (),
    model_kwargs: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Return sigmoid probabilities without modifying model parameters."""
    model.eval()
    output = model(*model_args, **dict(model_kwargs or {}))
    logits = output[0] if isinstance(output, tuple) else output
    return torch.sigmoid(logits.float())
