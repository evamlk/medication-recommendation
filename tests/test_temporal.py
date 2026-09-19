import numpy as np
import pandas as pd

from medrec.temporal import build_temporal_edges


def test_only_completed_history_is_connected() -> None:
    visits = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "visit_id": [10, 11, 12],
            "hadm_id": [100, 101, 102],
            "admittime": ["2024-01-01", "2024-01-05", "2024-01-20"],
            "dischtime": ["2024-01-10", "2024-01-08", "2024-01-22"],
        }
    )

    edges = build_temporal_edges(visits, max_history=2)

    np.testing.assert_array_equal(edges.edge_index, np.array([[11, 10], [12, 12]]))
    assert np.all((edges.edge_weight > 0) & (edges.edge_weight <= 1))


def test_empty_history_returns_well_shaped_arrays() -> None:
    visits = pd.DataFrame(
        {
            "subject_id": [1],
            "visit_id": [10],
            "hadm_id": [100],
            "admittime": ["2024-01-01"],
            "dischtime": ["2024-01-02"],
        }
    )

    edges = build_temporal_edges(visits)

    assert edges.edge_index.shape == (2, 0)
    assert edges.edge_weight.shape == (0,)
