"""Leakage-safe temporal graph construction for longitudinal admissions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalEdges:
    """Directed admission-history edges and their exponential-decay weights."""

    edge_index: np.ndarray
    edge_weight: np.ndarray


def build_temporal_edges(
    visits: pd.DataFrame,
    *,
    max_history: int = 1,
    decay_days: float = 180.0,
    allow_equal_discharge: bool = True,
) -> TemporalEdges:
    """Connect eligible completed admissions to later admissions.

    Required columns are ``subject_id``, ``visit_id``, ``hadm_id``,
    ``admittime``, and ``dischtime``. Edges are directed from a previous
    admission to a later admission for the same patient. A previous admission
    is eligible only when its discharge time is no later than the current
    admission time (strictly earlier when ``allow_equal_discharge`` is false).
    """
    required = {"subject_id", "visit_id", "hadm_id", "admittime", "dischtime"}
    missing = sorted(required.difference(visits.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if max_history < 1:
        raise ValueError("max_history must be at least one")
    if decay_days <= 0:
        raise ValueError("decay_days must be positive")
    if visits["visit_id"].duplicated().any():
        raise ValueError("visit_id values must be unique")

    frame = visits.copy()
    frame["admittime"] = pd.to_datetime(frame["admittime"], errors="raise")
    frame["dischtime"] = pd.to_datetime(frame["dischtime"], errors="coerce")

    sources: list[int] = []
    destinations: list[int] = []
    weights: list[float] = []

    for _, patient_visits in frame.groupby("subject_id", sort=False):
        ordered = patient_visits.sort_values(["admittime", "hadm_id"]).reset_index(drop=True)
        rows = list(ordered.itertuples(index=False))

        for current_position, current in enumerate(rows):
            current_time = pd.Timestamp(current.admittime)
            eligible = []
            for previous in rows[:current_position]:
                if pd.isna(previous.dischtime):
                    continue
                discharge_time = pd.Timestamp(previous.dischtime)
                is_eligible = (
                    discharge_time <= current_time
                    if allow_equal_discharge
                    else discharge_time < current_time
                )
                if is_eligible:
                    eligible.append(previous)

            eligible.sort(
                key=lambda row: (pd.Timestamp(row.admittime), int(row.hadm_id)),
                reverse=True,
            )
            for previous in eligible[:max_history]:
                gap_days = max(
                    (current_time - pd.Timestamp(previous.admittime)).total_seconds() / 86400,
                    0.0,
                )
                sources.append(int(previous.visit_id))
                destinations.append(int(current.visit_id))
                weights.append(float(np.exp(-gap_days / decay_days)))

    edge_index = np.asarray([sources, destinations], dtype=np.int64)
    if not sources:
        edge_index = np.empty((2, 0), dtype=np.int64)
    return TemporalEdges(edge_index=edge_index, edge_weight=np.asarray(weights, dtype=np.float32))
