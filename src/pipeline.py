from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ACCURACY_FILE,
    ACO_FEATURES,
    ACO_FEATURES_FLOOR,
    ACO_MODEL,
    CENTROID_CONFIDENCE_THRESHOLD,
    CENTROID_MODEL,
    EMBEDDING_SIZE,
    EXEMPLAR_DISTANCE_THRESHOLD,
    EXEMPLAR_MARGIN_RATIO,
    EXEMPLAR_MODEL,
    LOG_FILE,
    MLP_MODEL,
    PCA_COMPONENTS,
    RAW_IMAGE_MAX_DIM,
    PCA_MODEL,
    REGISTRATION_IMAGE_COUNT,
    RANDOM_STATE,
    SCALER_MODEL,
    SUPPORTED_IMAGE_EXTENSIONS,
    TEST_DIR,
    TRAINING_METADATA_MODEL,
    TRAIN_DIR,
    UNKNOWN_THRESHOLD,
    ensure_directories,
)
from database.db_operations import (
    get_registered_training_image_paths,
    log_prediction_event,
    register_criminal_with_images,
    store_unknown_face,
)
from src.aco_module import run_aco
from src.classifier import predict as classifier_predict
from src.classifier import predict_proba, train_classifier
from src.embedding import get_embedding, get_embedding_backend_name
from src.face_detection import DetectedFace, FaceBox, detect_face, detect_face_result
from src.pca_module import apply_feature_scaler, apply_pca, train_feature_scaler, train_pca
from src.preprocessing import load_image


ensure_directories()

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

EXEMPLAR_SUPPORT_THRESHOLD = 0.12
MLP_STANDALONE_THRESHOLD = 0.93


@dataclass
class PredictionResult:
    label: int | None
    confidence: float
    status: str
    used_fallback_face: bool = False
    face_box: tuple[int, int, int, int] | None = None


@dataclass
class VideoPredictionResult:
    label: int | None
    confidence: float
    status: str
    frames_processed: int
    matches: int


@dataclass
class RegistrationResult:
    criminal_id: int
    stored_images: list[str]
    train_samples: int
    methodology: str


class ModelArtifactsError(RuntimeError):
    pass


def describe_identification_flow() -> str:
    return (
        "Image/Video Input -> MTCNN Face Detection -> Face Alignment -> "
        "Preprocessing (160x160 normalized RGB) -> 512-D Embedding per image -> "
        "Feature Scaling -> PCA (up to 50-D) -> ACO (35-40-D) -> "
        "MLP Classification -> Database Matching -> GUI Display -> Logging/Storage"
    )


def parse_label_from_path(path: Path) -> int:
    candidates = []
    if path.parent.name.lower() not in {"train", "test"}:
        candidates.append(path.parent.name)
    candidates.append(path.stem)

    digits = ""
    for candidate in candidates:
        token = candidate.split("_")[0].split("-")[0]
        digits = "".join(character for character in token if character.isdigit())
        if digits:
            break
    if not digits:
        raise ValueError(f"Unable to infer label from path: {path}")
    return int(digits)


def collect_dataset(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    ensure_directories()
    dataset_path = Path(dataset_dir)
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []

    image_paths = _iter_training_ready_images(dataset_path)
    for image_path in image_paths:
        try:
            embedding, _ = build_feature_vector(image_path)
            label = parse_label_from_path(image_path)
        except Exception as error:
            logging.warning("Skipping %s due to %s", image_path, error)
            continue
        features.append(embedding)
        labels.append(label)
        paths.append(image_path)

    if not features:
        raise ValueError(f"No valid images were found inside {dataset_path}")

    return np.vstack(features), np.array(labels, dtype=int), paths


def build_feature_vector(
    image_path: str | Path,
    *,
    allow_missing_face: bool = False,
) -> tuple[np.ndarray, bool]:
    path = Path(image_path)
    is_preprocessed_face = _is_preprocessed_face_image(path)
    image = load_image(path) if is_preprocessed_face else load_image(path, max_dim=RAW_IMAGE_MAX_DIM)
    face = image if is_preprocessed_face else detect_face(image)
    if face is None:
        if not allow_missing_face:
            raise ValueError(f"No face detected in {path}")
        face = image
        used_fallback = True
    else:
        used_fallback = False
    embedding = get_embedding(face)
    return embedding.reshape(1, -1), used_fallback


def _iter_training_ready_images(dataset_path: Path) -> list[Path]:
    candidates = [
        image_path
        for image_path in sorted(dataset_path.rglob("*"))
        if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and not _is_legacy_rotation_variant(image_path)
    ]
    candidates = _drop_conflicting_duplicate_images(candidates)
    if dataset_path.resolve() == TRAIN_DIR.resolve():
        registered_paths = get_registered_training_image_paths()
        if registered_paths:
            candidates = _filter_registered_training_images(candidates, registered_paths)
    return candidates


def _is_legacy_rotation_variant(path: Path) -> bool:
    stem = path.stem.lower()
    return any(token in stem for token in ("rot90", "rot270", "flip_rot90", "debug"))


def _is_preprocessed_face_image(path: Path) -> bool:
    if path.suffix.lower() != ".png":
        return False
    if path.stem.lower().endswith(("_orig", "_flip")) and path.parent.name.lower().startswith("criminal_"):
        return True
    return False


def _drop_conflicting_duplicate_images(image_paths: list[Path]) -> list[Path]:
    grouped_paths: dict[str, list[Path]] = {}
    for image_path in image_paths:
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        grouped_paths.setdefault(digest, []).append(image_path)

    filtered_paths: list[Path] = []
    for paths in grouped_paths.values():
        labels = {parse_label_from_path(path) for path in paths}
        if len(labels) > 1:
            logging.warning("Skipping conflicting duplicate images across labels %s: %s", sorted(labels), paths)
            continue
        filtered_paths.extend(paths)
    return sorted(filtered_paths)


def _filter_registered_training_images(
    image_paths: list[Path],
    registered_paths: set[Path],
) -> list[Path]:
    if not registered_paths:
        return image_paths

    registered_lookup = {str(path.resolve()).lower() for path in registered_paths}
    filtered_paths = [
        path for path in image_paths if str(path.resolve()).lower() in registered_lookup
    ]
    skipped_paths = [path for path in image_paths if path not in filtered_paths]
    if skipped_paths:
        logging.warning(
            "Ignoring orphan training images that are not linked in criminal_images: %s",
            skipped_paths,
        )
    return filtered_paths


def train_pipeline(
    train_dir: str | Path = TRAIN_DIR,
    test_dir: str | Path | None = TEST_DIR,
) -> dict:
    ensure_directories()
    X_train, y_train, train_paths = collect_dataset(train_dir)
    X_train_scaled, scaler = train_feature_scaler(X_train, persist=True)
    X_pca, pca = train_pca(X_train_scaled, n_components=PCA_COMPONENTS, y=y_train, persist=True)
    selection_target = _resolve_aco_feature_count(X_pca.shape[1], y_train)
    selected_features = run_aco(
        X_pca,
        y_train,
        n_select=selection_target,
        random_state=RANDOM_STATE,
        persist=True,
    )
    X_selected = X_pca[:, selected_features]
    model = train_classifier(X_selected, y_train)
    centroid_model = build_centroid_model(X_selected, y_train)
    joblib.dump(centroid_model, CENTROID_MODEL)
    exemplar_model = build_exemplar_model(X_train, y_train, train_paths)
    joblib.dump(exemplar_model, EXEMPLAR_MODEL)

    metrics = {
        "train_samples": int(len(train_paths)),
        "classes": sorted(set(y_train.tolist())),
        "embedding_backend": get_embedding_backend_name(),
        "embedding_dim": EMBEDDING_SIZE,
        "scaled_dim": int(X_train_scaled.shape[1]),
        "pca_components": int(pca.n_components_),
        "aco_features": int(len(selected_features)),
        "sample_representation": "Each face image remains an independent sample vector.",
        "mlp_cv_balanced_accuracy": float(getattr(model, "cv_balanced_accuracy_", 0.0)),
    }
    TRAINING_METADATA_MODEL.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if test_dir and Path(test_dir).exists() and any(Path(test_dir).rglob("*")):
        X_test, y_test, _ = collect_dataset(test_dir)
        X_test_scaled = apply_feature_scaler(scaler, X_test)
        X_test_pca = apply_pca(pca, X_test_scaled)
        X_test_selected = X_test_pca[:, selected_features]
        predictions = classifier_predict(model, X_test_selected)
        accuracy = accuracy_score(y_test, predictions)
        metrics["test_accuracy"] = float(accuracy)
        ACCURACY_FILE.write_text(f"Accuracy: {accuracy:.4f}\n", encoding="utf-8")
        TRAINING_METADATA_MODEL.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    else:
        ACCURACY_FILE.write_text("Accuracy: test dataset not available\n", encoding="utf-8")

    logging.info("Training completed with %s samples.", len(train_paths))
    return metrics


def load_artifacts() -> tuple:
    try:
        return (
            joblib.load(SCALER_MODEL),
            joblib.load(PCA_MODEL),
            joblib.load(ACO_MODEL),
            joblib.load(MLP_MODEL),
            joblib.load(CENTROID_MODEL),
            joblib.load(EXEMPLAR_MODEL),
        )
    except Exception as error:
        raise ModelArtifactsError(str(error)) from error


def ensure_model_artifacts() -> tuple[Any, Any, Any, Any, Any, Any]:
    required_paths = (SCALER_MODEL, PCA_MODEL, ACO_MODEL, MLP_MODEL, CENTROID_MODEL, EXEMPLAR_MODEL)
    if not all(path.exists() for path in required_paths):
        logging.warning("Model artifacts missing. Training fresh models.")
        train_pipeline()
        return load_artifacts()

    try:
        return load_artifacts()
    except ModelArtifactsError as error:
        logging.warning("Model artifacts are stale or incompatible: %s", error)
        for artifact in required_paths:
            safe_unlink(artifact)
        train_pipeline()
        return load_artifacts()


def safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except FileNotFoundError:
        return


def persist_prediction_log(
    *,
    source_type: str,
    source_ref: str | Path | None,
    is_criminal: bool,
    predicted_label: int | None,
    confidence: float,
    status: str,
    embedding: np.ndarray,
    selected_features: np.ndarray,
) -> None:
    try:
        log_prediction_event(
            source_type=source_type,
            source_ref=source_ref,
            is_criminal=is_criminal,
            predicted_label=predicted_label,
            confidence=confidence,
            status=status,
            embedding=embedding,
            selected_features=selected_features,
        )
    except Exception as error:
        logging.warning("Unable to store prediction logs: %s", error)


def _face_box_tuple(face_box: FaceBox | None) -> tuple[int, int, int, int] | None:
    if face_box is None:
        return None
    return (int(face_box.x), int(face_box.y), int(face_box.width), int(face_box.height))


def _extract_face_for_prediction(image: np.ndarray) -> tuple[np.ndarray, bool, tuple[int, int, int, int] | None]:
    detection = detect_face_result(image)
    if detection is None:
        return image, True, None
    return detection.face, False, _face_box_tuple(detection.box)


def _predict_embedding_with_artifacts(
    embedding: np.ndarray,
    *,
    used_fallback: bool,
    face_box: tuple[int, int, int, int] | None,
    source_type: str,
    source_ref: str | Path | None,
    artifacts: tuple[Any, Any, Any, Any, Any, Any],
) -> PredictionResult:
    scaler, pca, selected_features, model, centroid_model, exemplar_model = artifacts
    scaled_embedding = apply_feature_scaler(scaler, embedding)
    transformed = apply_pca(pca, scaled_embedding)[:, selected_features]
    selected_feature_vector = transformed[0]
    probabilities = predict_proba(model, transformed)[0]
    best_index = int(np.argmax(probabilities))
    mlp_confidence = float(probabilities[best_index])
    mlp_label = int(model.classes_[best_index])
    exemplar_label, exemplar_confidence, exemplar_known = exemplar_verify(
        embedding[0],
        exemplar_model,
    )
    label, confidence, is_unknown = resolve_prediction(
        mlp_label=mlp_label,
        mlp_confidence=mlp_confidence,
        exemplar_label=exemplar_label,
        exemplar_confidence=exemplar_confidence,
        exemplar_known=exemplar_known,
        transformed_selected=selected_feature_vector,
        centroid_model=centroid_model,
    )

    if is_unknown or label is None:
        persist_prediction_log(
            source_type=source_type,
            source_ref=source_ref,
            is_criminal=False,
            predicted_label=None,
            confidence=confidence,
            status="Unknown Person",
            embedding=embedding[0],
            selected_features=selected_feature_vector,
        )
        return PredictionResult(
            label=None,
            confidence=confidence,
            status="Unknown Person",
            used_fallback_face=used_fallback,
            face_box=face_box,
        )

    persist_prediction_log(
        source_type=source_type,
        source_ref=source_ref,
        is_criminal=True,
        predicted_label=label,
        confidence=confidence,
        status="Criminal identified",
        embedding=embedding[0],
        selected_features=selected_feature_vector,
    )
    return PredictionResult(
        label=label,
        confidence=confidence,
        status="Criminal identified",
        used_fallback_face=used_fallback,
        face_box=face_box,
    )


def predict_image(image_path: str | Path) -> PredictionResult:
    ensure_directories()
    artifacts = ensure_model_artifacts()
    _, _, _, model, _, _ = artifacts
    if len(model.classes_) < 2:
        return PredictionResult(
            label=None,
            confidence=0.0,
            status=(
                "Only one registered criminal is in the training data. "
                "Add more registered criminals for reliable identification."
            ),
            used_fallback_face=False,
        )

    image = load_image(image_path, max_dim=RAW_IMAGE_MAX_DIM)
    face, used_fallback, face_box = _extract_face_for_prediction(image)
    embedding = get_embedding(face).reshape(1, -1)
    result = _predict_embedding_with_artifacts(
        embedding,
        used_fallback=used_fallback,
        face_box=face_box,
        source_type="image",
        source_ref=image_path,
        artifacts=artifacts,
    )
    if result.label is None:
        unknown_path = store_unknown_face(image_path)
        logging.info("Unknown face stored at %s", unknown_path)
    return result


def predict_frame(
    frame: np.ndarray,
    source_type: str = "video",
    source_ref: str | Path | None = None,
    *,
    artifacts: tuple[Any, Any, Any, Any, Any, Any] | None = None,
) -> PredictionResult:
    ensure_directories()
    artifacts = artifacts or ensure_model_artifacts()
    _, _, _, model, _, _ = artifacts
    if len(model.classes_) < 2:
        return PredictionResult(
            label=None,
            confidence=0.0,
            status=(
                "Only one registered criminal is in the training data. "
                "Video identification is not reliable yet."
            ),
            used_fallback_face=False,
        )

    face, used_fallback, face_box = _extract_face_for_prediction(frame)
    embedding = get_embedding(face).reshape(1, -1)
    return _predict_embedding_with_artifacts(
        embedding,
        used_fallback=used_fallback,
        face_box=face_box,
        source_type=source_type,
        source_ref=source_ref,
        artifacts=artifacts,
    )


def identify_from_video(
    video_path: str | Path,
    frame_interval: int = 10,
    max_frames: int = 120,
    on_frame=None,
) -> VideoPredictionResult:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is required for video identification.") from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    artifacts = ensure_model_artifacts()
    frame_index = 0
    processed = 0
    votes: dict[int, list[float]] = {}
    unknown_votes = 0
    try:
        while processed < max_frames:
            success, frame = capture.read()
            if not success:
                break
            frame_index += 1
            if frame_index % frame_interval != 0:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = predict_frame(
                rgb_frame,
                source_type="video",
                source_ref=str(video_path),
                artifacts=artifacts,
            )
            processed += 1
            if on_frame is not None:
                on_frame(rgb_frame.copy(), prediction, processed, frame_index)
            if prediction.label is None:
                unknown_votes += 1
                continue
            votes.setdefault(prediction.label, []).append(prediction.confidence)
    finally:
        capture.release()

    if not votes:
        return VideoPredictionResult(
            label=None,
            confidence=0.0,
            status="Unknown Person",
            frames_processed=processed,
            matches=0,
        )

    best_label, confidences = max(
        votes.items(),
        key=lambda item: (len(item[1]), float(np.mean(item[1]))),
    )
    mean_confidence = float(np.mean(confidences))
    if unknown_votes >= len(confidences):
        return VideoPredictionResult(
            label=None,
            confidence=mean_confidence,
            status="Unknown Person",
            frames_processed=processed,
            matches=len(confidences),
        )

    return VideoPredictionResult(
        label=best_label,
        confidence=mean_confidence,
        status="Criminal identified from video",
        frames_processed=processed,
        matches=len(confidences),
    )


def register_criminal_pipeline(
    *,
    name: str,
    dob: str | None,
    moles: str,
    nationality: str,
    region: str,
    crime: str,
    num_crimes: int,
    image_paths: list[str],
) -> RegistrationResult:
    criminal_id, stored_images = register_criminal_with_images(
        name=name,
        dob=dob,
        moles=moles,
        nationality=nationality,
        region=region,
        crime=crime,
        num_crimes=num_crimes,
        image_paths=image_paths,
    )
    metrics = train_pipeline()
    methodology = (
        f"Registered with {REGISTRATION_IMAGE_COUNT} images. "
        "Overall flow: "
        "Each saved face image becomes its own training sample. "
        "Image/Video Input -> Face Detection -> Face Alignment -> "
        "Preprocessing (160x160 normalized RGB) -> "
        f"{get_embedding_backend_name()} 512-D Embedding per image -> "
        "Feature Scaling -> PCA (up to 50-D) -> ACO (35-40-D) -> "
        "MLP Classification -> Database Matching -> GUI Display -> Logging/Storage."
    )
    return RegistrationResult(
        criminal_id=criminal_id,
        stored_images=stored_images,
        train_samples=metrics["train_samples"],
        methodology=methodology,
    )


def build_centroid_model(X_scaled: np.ndarray, y: np.ndarray) -> dict:
    centroids: dict[int, np.ndarray] = {}
    thresholds: dict[int, float] = {}
    for label in sorted(set(y.tolist())):
        class_samples = X_scaled[y == label]
        centroid = class_samples.mean(axis=0)
        centroids[int(label)] = centroid
        distances = np.linalg.norm(class_samples - centroid, axis=1)
        thresholds[int(label)] = float(distances.max() + max(distances.std(), 0.15))
    return {"centroids": centroids, "thresholds": thresholds}


def build_exemplar_model(X: np.ndarray, y: np.ndarray, paths: list[Path]) -> dict:
    distance_limit, margin_ratio = _estimate_exemplar_thresholds(X, y)
    return {
        "embeddings": np.asarray(X, dtype="float32"),
        "labels": np.asarray(y, dtype=int),
        "paths": [str(path) for path in paths],
        "distance_limit": distance_limit,
        "margin_ratio": margin_ratio,
    }


def exemplar_verify(
    sample_embedding: np.ndarray,
    exemplar_model: dict,
) -> tuple[int | None, float, bool]:
    embeddings = np.asarray(exemplar_model["embeddings"], dtype="float32")
    labels = np.asarray(exemplar_model["labels"], dtype=int)
    distance_limit = float(exemplar_model.get("distance_limit", EXEMPLAR_DISTANCE_THRESHOLD))
    margin_ratio = float(exemplar_model.get("margin_ratio", EXEMPLAR_MARGIN_RATIO))
    if embeddings.size == 0 or labels.size == 0:
        return None, 0.0, False

    distances = np.linalg.norm(embeddings - sample_embedding.reshape(1, -1), axis=1)
    class_best: dict[int, float] = {}
    for label, distance in zip(labels.tolist(), distances.tolist()):
        best_distance = class_best.get(int(label))
        if best_distance is None or distance < best_distance:
            class_best[int(label)] = float(distance)

    ordered = sorted(class_best.items(), key=lambda item: item[1])
    if not ordered:
        return None, 0.0, False

    best_label, best_distance = ordered[0]
    second_distance = ordered[1][1] if len(ordered) > 1 else float("inf")
    is_known = (
        best_distance <= distance_limit
        and best_distance <= second_distance * margin_ratio
    )
    confidence = float(
        np.clip(1.0 - (best_distance / max(distance_limit, 1e-6)), 0.0, 1.0)
    )
    return (best_label if is_known else None), confidence, is_known


def resolve_prediction(
    *,
    mlp_label: int,
    mlp_confidence: float,
    exemplar_label: int | None,
    exemplar_confidence: float,
    exemplar_known: bool,
    transformed_selected: np.ndarray,
    centroid_model: dict,
) -> tuple[int | None, float, bool]:
    centroid_label, centroid_confidence, centroid_unknown = centroid_verify(
        transformed_selected,
        centroid_model,
        fallback_label=mlp_label,
        fallback_confidence=mlp_confidence,
    )

    strong_exemplar = (
        exemplar_known
        and exemplar_label is not None
        and exemplar_confidence >= EXEMPLAR_SUPPORT_THRESHOLD
    )
    strong_centroid = (
        not centroid_unknown
        and centroid_label is not None
        and centroid_confidence >= CENTROID_CONFIDENCE_THRESHOLD
    )

    support: dict[int, list[float]] = {mlp_label: [mlp_confidence]}
    if strong_exemplar and exemplar_label is not None:
        support.setdefault(exemplar_label, []).append(exemplar_confidence)
        if exemplar_label != mlp_label and exemplar_confidence >= max(0.55, mlp_confidence + 0.08):
            return exemplar_label, exemplar_confidence, False
    if strong_centroid and centroid_label is not None:
        support.setdefault(centroid_label, []).append(centroid_confidence)

    if strong_exemplar and exemplar_label == mlp_label:
        return mlp_label, float(max(mlp_confidence, exemplar_confidence)), False
    if strong_centroid and centroid_label == mlp_label:
        return mlp_label, float(max(mlp_confidence, centroid_confidence)), False
    if strong_exemplar and strong_centroid and exemplar_label == centroid_label:
        return exemplar_label, float(max(exemplar_confidence, centroid_confidence)), False

    best_label = max(
        support,
        key=lambda label: (len(support[label]), float(np.mean(support[label])), float(max(support[label]))),
    )
    votes = len(support[best_label])
    best_confidence = float(max(support[best_label]))
    if votes >= 2 and best_confidence >= UNKNOWN_THRESHOLD:
        return best_label, best_confidence, False
    if mlp_confidence >= MLP_STANDALONE_THRESHOLD:
        return mlp_label, mlp_confidence, False
    if strong_exemplar and exemplar_label is not None and exemplar_confidence >= UNKNOWN_THRESHOLD - 0.08:
        return exemplar_label, exemplar_confidence, False
    if strong_centroid and centroid_label is not None and centroid_confidence >= UNKNOWN_THRESHOLD - 0.08:
        return centroid_label, centroid_confidence, False
    return None, max(best_confidence, centroid_confidence), True


def centroid_verify(
    sample_selected: np.ndarray,
    centroid_model: dict,
    fallback_label: int,
    fallback_confidence: float,
) -> tuple[int | None, float, bool]:
    centroids = centroid_model["centroids"]
    thresholds = centroid_model["thresholds"]
    labels = sorted(centroids.keys())
    centroid_matrix = np.vstack([centroids[label] for label in labels])
    similarities = cosine_similarity(sample_selected.reshape(1, -1), centroid_matrix)[0]
    distances = np.linalg.norm(centroid_matrix - sample_selected.reshape(1, -1), axis=1)

    best_idx = int(np.argmax(similarities))
    centroid_label = int(labels[best_idx])
    centroid_similarity = float(similarities[best_idx])
    centroid_distance = float(distances[best_idx])
    distance_limit = float(thresholds[centroid_label])

    if centroid_distance > distance_limit and centroid_similarity < CENTROID_CONFIDENCE_THRESHOLD:
        return None, centroid_similarity, True

    is_strong_match = (
        centroid_similarity >= CENTROID_CONFIDENCE_THRESHOLD
        or (
            centroid_similarity >= CENTROID_CONFIDENCE_THRESHOLD - 0.08
            and centroid_distance <= distance_limit * 0.75
        )
    )
    if not is_strong_match:
        return None, centroid_similarity, True

    if centroid_label == fallback_label or centroid_similarity >= max(fallback_confidence - 0.1, CENTROID_CONFIDENCE_THRESHOLD):
        return centroid_label, centroid_similarity, False

    return None, centroid_similarity, True


def _resolve_aco_feature_count(n_features: int, y: np.ndarray) -> int:
    class_count = max(1, len(np.unique(y)))
    if n_features <= ACO_FEATURES_FLOOR:
        return n_features
    if n_features <= ACO_FEATURES:
        return max(ACO_FEATURES_FLOOR, n_features)
    selection_target = max(class_count * 8, int(round(n_features * 0.7)))
    selection_target = max(ACO_FEATURES_FLOOR, selection_target)
    return max(1, min(ACO_FEATURES, n_features, selection_target))


def _estimate_exemplar_thresholds(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(X) < 3 or len(np.unique(y)) < 2:
        return float(EXEMPLAR_DISTANCE_THRESHOLD), float(EXEMPLAR_MARGIN_RATIO)

    nearest_same: list[float] = []
    nearest_other: list[float] = []
    for index, label in enumerate(y.tolist()):
        sample = X[index]
        distances = np.linalg.norm(X - sample.reshape(1, -1), axis=1)
        same_mask = y == label
        same_mask[index] = False
        other_mask = ~same_mask
        other_mask[index] = False
        if np.any(same_mask):
            nearest_same.append(float(np.min(distances[same_mask])))
        if np.any(other_mask):
            nearest_other.append(float(np.min(distances[other_mask])))

    if not nearest_same or not nearest_other:
        return float(EXEMPLAR_DISTANCE_THRESHOLD), float(EXEMPLAR_MARGIN_RATIO)

    same_array = np.asarray(nearest_same, dtype="float32")
    other_array = np.asarray(nearest_other, dtype="float32")
    distance_limit = float(np.percentile(same_array, 95))
    ratio = np.divide(
        same_array,
        np.maximum(other_array[: len(same_array)], 1e-6),
    )
    margin_ratio = float(np.clip(np.percentile(ratio, 90) * 1.05, 0.2, 0.95))
    return max(distance_limit, 1e-6), margin_ratio
