import numpy as np

from src.pca_module import apply_pca, train_pca


def test_train_pca_reduces_features():
    X = np.random.default_rng(42).random((8, 16))
    X_pca, pca = train_pca(X, n_components=5)

    assert X_pca.shape == (8, pca.n_components_)
    assert 1 <= pca.n_components_ <= 5
    assert np.allclose(apply_pca(pca, X), X_pca)
