from __future__ import annotations

import numpy as np
import cv2

from config import EMBEDDING_SIZE, IMAGE_SIZE
from src.facenet_runtime import extract_facenet_embedding, get_facenet_backend_status
from src.preprocessing import convert_to_grayscale, prepare_face_image


def get_embedding(face: np.ndarray) -> np.ndarray:
    facenet_embedding = extract_facenet_embedding(face)
    if facenet_embedding is not None:
        return facenet_embedding
    return get_fallback_embedding(face)


def get_fallback_embedding(face: np.ndarray) -> np.ndarray:
    # Build the embedding directly from the normalized 160x160 RGB face image
    # so registration and identification always describe the same face crop.
    # The descriptor is intentionally structured to stay 512-D:
    # 196 low-frequency + 128 HOG + 100 DCT + 40 LBP + 36 HSV + 12 stats.
    color_face = np.clip(prepare_face_image(face, size=IMAGE_SIZE) * 255.0, 0, 255).astype("uint8")
    grayscale_u8 = convert_to_grayscale(color_face)
    equalized = cv2.equalizeHist(grayscale_u8)

    low_frequency = (
        cv2.resize(equalized, (14, 14), interpolation=cv2.INTER_AREA).astype("float32").reshape(-1) / 255.0
    )
    gradient_features = _gradient_histograms(equalized, grid=(4, 4), bins=8)

    dct_source = cv2.resize(equalized, (32, 32), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    dct_features = cv2.dct(dct_source)[:10, :10].reshape(-1)
    dct_features = _normalize_vector(dct_features)

    # LBP-style histogram contributes 40 dimensions.
    lbp = _local_binary_pattern(equalized)
    lbp_hist, _ = np.histogram(lbp // 6, bins=40, range=(0, 40), density=True)

    hsv_face = cv2.cvtColor(color_face, cv2.COLOR_RGB2HSV)
    color_histograms = [
        np.histogram(hsv_face[:, :, channel], bins=12, range=(0, 256), density=True)[0]
        for channel in range(3)
    ]
    color_features = np.concatenate(color_histograms).astype("float32")

    # Global face statistics contributes 12 dimensions.
    edge_map = cv2.Canny(equalized, 50, 150)
    stats = np.array(
        [
            color_face[:, :, 0].mean(),
            color_face[:, :, 1].mean(),
            color_face[:, :, 2].mean(),
            color_face[:, :, 0].std(),
            color_face[:, :, 1].std(),
            color_face[:, :, 2].std(),
            equalized.mean(),
            equalized.std(),
            edge_map.mean(),
            edge_map.std(),
            hsv_face[:, :, 1].mean(),
            hsv_face[:, :, 1].std(),
            hsv_face[:, :, 2].mean(),
            hsv_face[:, :, 2].std(),
        ],
        dtype="float32",
    )
    stats /= 255.0

    embedding = np.concatenate(
        [
            low_frequency.astype("float32"),
            gradient_features.astype("float32"),
            dct_features.astype("float32"),
            lbp_hist.astype("float32"),
            color_features,
            stats,
        ]
    )
    embedding = embedding[:EMBEDDING_SIZE].astype("float32")
    if embedding.size < EMBEDDING_SIZE:
        embedding = np.pad(embedding, (0, EMBEDDING_SIZE - embedding.size))
    return _normalize_vector(embedding.astype("float32"))


def get_embedding_backend_name() -> str:
    status = get_facenet_backend_status()
    if status.available:
        return "FaceNet"
    return "FallbackDescriptor"


def compare_faces(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    return float(np.linalg.norm(embedding_a - embedding_b))


def is_same_person(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    threshold: float = 0.35,
) -> tuple[bool, float]:
    distance = compare_faces(embedding_a, embedding_b)
    return distance <= threshold, distance


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype="float32").reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return vector


def _local_binary_pattern(image: np.ndarray) -> np.ndarray:
    center = image[1:-1, 1:-1]
    result = np.zeros_like(center, dtype=np.uint8)
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = image[1 + dy : image.shape[0] - 1 + dy, 1 + dx : image.shape[1] - 1 + dx]
        result |= ((neighbor >= center) << bit).astype(np.uint8)
    return result


def _gradient_histograms(
    image: np.ndarray,
    *,
    grid: tuple[int, int],
    bins: int,
) -> np.ndarray:
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle %= 180.0

    rows, cols = grid
    cell_height = max(1, image.shape[0] // rows)
    cell_width = max(1, image.shape[1] // cols)
    histograms: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            y_start = row * cell_height
            y_end = image.shape[0] if row == rows - 1 else (row + 1) * cell_height
            x_start = col * cell_width
            x_end = image.shape[1] if col == cols - 1 else (col + 1) * cell_width
            cell_angles = angle[y_start:y_end, x_start:x_end].reshape(-1)
            cell_magnitude = magnitude[y_start:y_end, x_start:x_end].reshape(-1)
            histogram, _ = np.histogram(
                cell_angles,
                bins=bins,
                range=(0.0, 180.0),
                weights=cell_magnitude,
                density=False,
            )
            histogram = histogram.astype("float32")
            total = float(histogram.sum())
            if total > 0.0:
                histogram /= total
            histograms.append(histogram)
    return np.concatenate(histograms)
