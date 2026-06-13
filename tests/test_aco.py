import numpy as np

from src.aco_module import run_aco


def test_run_aco_returns_valid_indices():
    rng = np.random.default_rng(7)
    X = rng.random((10, 12))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    selected = run_aco(X, y, n_ants=4, n_iter=5, n_select=6, random_state=1)

    assert len(selected) == 6
    assert np.all(selected >= 0)
    assert np.all(selected < 12)
