import numpy as np

from src.classifier import predict_proba, train_classifier


def test_train_classifier_produces_probabilities() -> None:
    rng = np.random.default_rng(42)
    class_a = rng.normal(loc=-2.0, scale=0.2, size=(8, 6))
    class_b = rng.normal(loc=0.0, scale=0.2, size=(8, 6))
    class_c = rng.normal(loc=2.0, scale=0.2, size=(8, 6))
    X = np.vstack([class_a, class_b, class_c]).astype("float32")
    y = np.array([1] * 8 + [2] * 8 + [3] * 8, dtype=int)

    model = train_classifier(X, y)
    probabilities = predict_proba(model, X[:3])

    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_train_classifier_handles_single_class() -> None:
    X = np.ones((5, 4), dtype="float32")
    y = np.array([7] * 5, dtype=int)

    model = train_classifier(X, y)
    predictions = model.predict(X[:2])
    probabilities = predict_proba(model, X[:2])

    assert np.array_equal(predictions, np.array([7, 7]))
    assert probabilities.shape == (2, 1)
    assert np.allclose(probabilities, 1.0)
