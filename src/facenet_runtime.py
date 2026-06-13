from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading

import numpy as np

from config import FACENET_MODEL, FACENET_MODEL_DIR, IMAGE_SIZE
from src.preprocessing import ensure_rgb, resize_image

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass(frozen=True)
class FaceNetBackendStatus:
    available: bool
    model_path: Path | None
    reason: str


class FaceNetRuntime:
    def __init__(self, model_path: Path) -> None:
        try:
            import tensorflow.compat.v1 as tf  # type: ignore
        except ModuleNotFoundError as error:  # pragma: no cover
            raise RuntimeError("TensorFlow is not installed for FaceNet inference.") from error

        tf.disable_eager_execution()
        self._tf = tf
        self.model_path = model_path
        self.graph = tf.Graph()
        saver = None
        checkpoint_path = None
        with self.graph.as_default():
            if model_path.is_file():
                self._load_graph_def(model_path)
            else:
                saver, checkpoint_path = self._prepare_checkpoint_restore(model_path)

        self.session = tf.Session(graph=self.graph)
        if saver is not None and checkpoint_path is not None:
            saver.restore(self.session, checkpoint_path)
        self.input_tensor = self.graph.get_tensor_by_name("input:0")
        self.embedding_tensor = self.graph.get_tensor_by_name("embeddings:0")
        self.phase_train_tensor = self._get_optional_tensor("phase_train:0")

    def embed(self, face: np.ndarray) -> np.ndarray:
        image = prepare_facenet_tensor(face)
        feed_dict = {self.input_tensor: image[np.newaxis, ...]}
        if self.phase_train_tensor is not None:
            feed_dict[self.phase_train_tensor] = False
        embedding = self.session.run(self.embedding_tensor, feed_dict=feed_dict)[0]
        embedding = np.asarray(embedding, dtype="float32").reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding = embedding / norm
        return embedding

    def _load_graph_def(self, model_path: Path) -> None:
        with self._tf.gfile.FastGFile(str(model_path), "rb") as model_file:
            graph_def = self._tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            self._tf.import_graph_def(graph_def, name="")

    def _prepare_checkpoint_restore(self, model_dir: Path):
        meta_files = sorted(model_dir.glob("*.meta"))
        if len(meta_files) != 1:
            raise RuntimeError(f"Expected exactly one FaceNet .meta file in {model_dir}")

        checkpoint = self._tf.train.get_checkpoint_state(str(model_dir))
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_path = checkpoint.model_checkpoint_path
        else:
            checkpoint_candidates = sorted(model_dir.glob("*.ckpt-*.index"))
            if not checkpoint_candidates:
                raise RuntimeError(f"Unable to find FaceNet checkpoint files in {model_dir}")
            checkpoint_path = str(checkpoint_candidates[-1]).rsplit(".index", 1)[0]

        saver = self._tf.train.import_meta_graph(str(meta_files[0]))
        return saver, checkpoint_path

    def _get_optional_tensor(self, tensor_name: str):
        try:
            return self.graph.get_tensor_by_name(tensor_name)
        except KeyError:
            return None


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: FaceNetRuntime | None = None
_RUNTIME_FAILURE: str | None = None


def facenet_prewhiten(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype="float32")
    mean = np.mean(array)
    std = np.std(array)
    std_adj = max(float(std), 1.0 / float(np.sqrt(array.size)))
    return (array - mean) / std_adj


def prepare_facenet_tensor(face: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    rgb_face = ensure_rgb(face)
    resized = resize_image(rgb_face, size=size).astype("float32")
    return facenet_prewhiten(resized)


def get_facenet_backend_status() -> FaceNetBackendStatus:
    model_path = resolve_facenet_model_path()
    if model_path is None:
        return FaceNetBackendStatus(
            available=False,
            model_path=None,
            reason="No pretrained FaceNet .pb file or checkpoint directory was found.",
        )

    try:
        import tensorflow.compat.v1  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        return FaceNetBackendStatus(
            available=False,
            model_path=model_path,
            reason="TensorFlow is not installed for the FaceNet model runtime.",
        )

    return FaceNetBackendStatus(
        available=True,
        model_path=model_path,
        reason="Pretrained FaceNet runtime is ready.",
    )


def extract_facenet_embedding(face: np.ndarray) -> np.ndarray | None:
    runtime = _get_runtime()
    if runtime is None:
        return None
    return runtime.embed(face)


def resolve_facenet_model_path() -> Path | None:
    if FACENET_MODEL.exists():
        return FACENET_MODEL
    if FACENET_MODEL_DIR.exists():
        pb_files = sorted(FACENET_MODEL_DIR.glob("*.pb"))
        if pb_files:
            return pb_files[0]
        if list(FACENET_MODEL_DIR.glob("*.meta")):
            return FACENET_MODEL_DIR
    return None


def get_facenet_runtime_failure() -> str | None:
    return _RUNTIME_FAILURE


def _get_runtime() -> FaceNetRuntime | None:
    global _RUNTIME
    global _RUNTIME_FAILURE

    status = get_facenet_backend_status()
    if not status.available or status.model_path is None:
        _RUNTIME_FAILURE = status.reason
        return None

    if _RUNTIME is not None:
        return _RUNTIME

    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME
        try:
            _RUNTIME = FaceNetRuntime(status.model_path)
            _RUNTIME_FAILURE = None
        except Exception as error:  # pragma: no cover
            _RUNTIME_FAILURE = str(error)
            _RUNTIME = None
    return _RUNTIME
