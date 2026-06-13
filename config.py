from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
MODEL_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
OUTPUT_DIR = BASE_DIR / "outputs"
UNKNOWN_FACE_DIR = OUTPUT_DIR / "unknown_faces"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"

PCA_MODEL = MODEL_DIR / "pca_model.pkl"
ACO_MODEL = MODEL_DIR / "aco_features.pkl"
MLP_MODEL = MODEL_DIR / "mlp_model.pkl"
SCALER_MODEL = MODEL_DIR / "scaler.pkl"
CENTROID_MODEL = MODEL_DIR / "centroid_model.pkl"
EXEMPLAR_MODEL = MODEL_DIR / "exemplar_model.pkl"
FACENET_MODEL = MODEL_DIR / "facenet_model.pb"
FACENET_MODEL_DIR = MODEL_DIR / "facenet"
TRAINING_METADATA_MODEL = MODEL_DIR / "training_metadata.json"

DATABASE_PATH = DATABASE_DIR / "criminal_identification.db"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
ACCURACY_FILE = RESULTS_DIR / "accuracy.txt"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
LOG_FILE = LOG_DIR / "system.log"

IMAGE_SIZE = (160, 160)
EMBEDDING_SIZE = 512
DETECTION_MAX_DIM = 960
RAW_IMAGE_MAX_DIM = 1600
PCA_COMPONENTS = 50
ACO_FEATURES = 40
ACO_FEATURES_FLOOR = 35
MLP_HIDDEN_LAYER_SIZES = (128, 64)
MLP_MAX_ITERATIONS = 600
RANDOM_STATE = 42
UNKNOWN_THRESHOLD = 0.58
CENTROID_CONFIDENCE_THRESHOLD = 0.52
EXEMPLAR_DISTANCE_THRESHOLD = 8.0
EXEMPLAR_MARGIN_RATIO = 0.72
REGISTRATION_IMAGE_COUNT = 5
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path = BASE_DIR
    dataset_dir: Path = DATASET_DIR
    train_dir: Path = TRAIN_DIR
    test_dir: Path = TEST_DIR
    model_dir: Path = MODEL_DIR
    database_dir: Path = DATABASE_DIR
    output_dir: Path = OUTPUT_DIR
    unknown_face_dir: Path = UNKNOWN_FACE_DIR
    results_dir: Path = RESULTS_DIR
    log_dir: Path = LOG_DIR
    training_metadata_model: Path = TRAINING_METADATA_MODEL
    database_path: Path = DATABASE_PATH
    schema_path: Path = SCHEMA_PATH
    accuracy_file: Path = ACCURACY_FILE
    roc_curve_path: Path = ROC_CURVE_PATH
    log_file: Path = LOG_FILE


PATHS = ProjectPaths()


def ensure_directories() -> None:
    for directory in (
        MODEL_DIR,
        FACENET_MODEL_DIR,
        DATABASE_DIR,
        OUTPUT_DIR,
        UNKNOWN_FACE_DIR,
        RESULTS_DIR,
        LOG_DIR,
        DATASET_DIR,
        TRAIN_DIR,
        TEST_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
