from __future__ import annotations

import numpy as np

from database.db_operations import _generate_face_variants
import src.face_detection as face_detection
from src.preprocessing import prepare_facenet_input


def test_prepare_facenet_input_returns_normalized_rgb() -> None:
    grayscale = np.full((18, 18), 120, dtype=np.uint8)

    prepared = prepare_facenet_input(grayscale, size=(32, 32))

    assert prepared.shape == (32, 32, 3)
    assert prepared.dtype == np.float32
    assert float(prepared.min()) >= 0.0
    assert float(prepared.max()) <= 1.0


def test_generate_face_variants_keeps_horizontal_flip_only() -> None:
    face = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    variants = _generate_face_variants(face)

    assert [name for name, _ in variants] == ["orig", "flip"]
    assert np.array_equal(variants[0][1], face)
    assert np.array_equal(variants[1][1], np.fliplr(face))


def test_detect_face_aligns_the_selected_crop(monkeypatch) -> None:
    image = np.arange(5 * 5 * 3, dtype=np.uint8).reshape(5, 5, 3)

    monkeypatch.setattr(
        face_detection,
        "_detect_face_detections",
        lambda current_image: [
            face_detection.FaceDetection(
                box=face_detection.FaceBox(1, 1, 2, 2),
                confidence=0.9,
                keypoints={"left_eye": (1.2, 1.4), "right_eye": (2.4, 1.5)},
                source="mtcnn",
            )
        ],
    )
    monkeypatch.setattr(
        face_detection,
        "_expand_face_box",
        lambda face, image_shape, margin_ratio: face,
    )
    monkeypatch.setattr(
        face_detection,
        "_refine_face_crop",
        lambda crop: crop,
    )
    monkeypatch.setattr(
        face_detection,
        "_normalize_face_orientation_with_keypoints",
        lambda crop, keypoints: (crop, keypoints),
    )
    monkeypatch.setattr(
        face_detection,
        "_align_face_with_keypoints",
        lambda crop, keypoints: crop,
    )
    monkeypatch.setattr(
        face_detection,
        "_normalize_face_orientation",
        lambda crop: crop,
    )
    monkeypatch.setattr(
        face_detection,
        "_align_face",
        lambda crop: crop + 1,
    )

    detected = face_detection.detect_face(image)

    assert np.array_equal(detected, image[1:3, 1:3] + 1)


def test_select_best_detection_prefers_landmark_rich_mtcnn_when_areas_are_close() -> None:
    detections = [
        face_detection.FaceDetection(
            box=face_detection.FaceBox(0, 0, 50, 50),
            confidence=0.55,
            keypoints=None,
            source="haar",
        ),
        face_detection.FaceDetection(
            box=face_detection.FaceBox(0, 0, 48, 52),
            confidence=0.92,
            keypoints={
                "left_eye": (12.0, 18.0),
                "right_eye": (34.0, 17.5),
                "nose": (23.0, 28.0),
                "mouth_left": (16.0, 37.0),
                "mouth_right": (31.0, 37.5),
            },
            source="mtcnn",
        ),
    ]

    selected = face_detection._select_best_detection(detections)

    assert selected.source == "mtcnn"
    assert selected.confidence == 0.92


def test_expand_face_box_adds_margin_without_leaving_image_bounds() -> None:
    face = face_detection.FaceBox(10, 12, 20, 24)

    expanded = face_detection._expand_face_box(
        face,
        image_shape=(40, 50, 3),
        margin_ratio=0.2,
    )

    assert expanded.x < face.x
    assert expanded.y < face.y
    assert expanded.width > face.width
    assert expanded.height > face.height
    assert expanded.x >= 0
    assert expanded.y >= 0
    assert expanded.x + expanded.width <= 50
    assert expanded.y + expanded.height <= 40


def test_normalize_face_orientation_rotates_when_rotated_score_is_better(monkeypatch) -> None:
    image = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    rotated = np.rot90(image, 2).copy()

    def fake_score(current_image: np.ndarray) -> float | None:
        if np.array_equal(current_image, image):
            return None
        if np.array_equal(current_image, rotated):
            return 1.0
        return None

    monkeypatch.setattr(face_detection, "_upright_orientation_score", fake_score)

    normalized = face_detection._normalize_face_orientation(image)

    assert np.array_equal(normalized, rotated)


def test_normalize_face_orientation_can_choose_quarter_turn(monkeypatch) -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    rotated = np.rot90(image, 1).copy()

    def fake_score(current_image: np.ndarray) -> float | None:
        if np.array_equal(current_image, rotated):
            return 1.0
        if np.array_equal(current_image, image):
            return 0.1
        return None

    monkeypatch.setattr(face_detection, "_upright_orientation_score", fake_score)

    normalized = face_detection._normalize_face_orientation(image)

    assert np.array_equal(normalized, rotated)


def test_normalize_face_orientation_keeps_image_when_scores_are_missing(monkeypatch) -> None:
    image = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)

    monkeypatch.setattr(face_detection, "_upright_orientation_score", lambda current_image: None)

    normalized = face_detection._normalize_face_orientation(image)

    assert np.array_equal(normalized, image)


def test_resize_for_detection_scales_down_large_images() -> None:
    image = np.zeros((4000, 3000, 3), dtype=np.uint8)

    resized, scale_x, scale_y = face_detection._resize_for_detection(image)

    assert max(resized.shape[:2]) <= 960
    assert scale_x > 1.0
    assert scale_y > 1.0
