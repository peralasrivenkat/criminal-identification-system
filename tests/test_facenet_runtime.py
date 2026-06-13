import numpy as np

import src.facenet_runtime as facenet_runtime


def test_facenet_prewhiten_centers_the_tensor() -> None:
    image = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

    whitened = facenet_runtime.facenet_prewhiten(image)

    assert whitened.shape == image.shape
    assert abs(float(np.mean(whitened))) < 1e-5
    assert float(np.std(whitened)) > 0.99


def test_backend_status_reports_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(facenet_runtime, "resolve_facenet_model_path", lambda: None)

    status = facenet_runtime.get_facenet_backend_status()

    assert status.available is False
    assert status.model_path is None
    assert "No pretrained FaceNet" in status.reason
