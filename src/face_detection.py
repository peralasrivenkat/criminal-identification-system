from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from config import DETECTION_MAX_DIM

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class FaceDetection:
    box: FaceBox
    confidence: float = 0.0
    keypoints: dict[str, tuple[float, float]] | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class FaceCandidate:
    detection: FaceDetection
    area: int
    rotation: int


@dataclass(frozen=True)
class DetectedFace:
    face: np.ndarray
    box: FaceBox
    confidence: float = 0.0
    source: str = "unknown"
    rotation: int = 0


MTCNN_MIN_CONFIDENCE = 0.50
PRIMARY_CROP_MARGIN_RATIO = 0.18
REFINEMENT_CROP_MARGIN_RATIO = 0.12


CASCADE = None
EYE_CASCADE = None
if cv2 is not None:  # pragma: no branch
    CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    EYE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )

_MTCNN_DETECTOR = None
_MTCNN_IMPORT_ATTEMPTED = False


def detect_faces(image: np.ndarray) -> list[FaceBox]:
    return [detection.box for detection in _detect_face_detections(image)]


def _detect_face_detections(image: np.ndarray) -> list[FaceDetection]:
    mtcnn_faces = detect_faces_mtcnn(image)
    if mtcnn_faces:
        return mtcnn_faces
    return detect_faces_haar(image)


def detect_faces_mtcnn(image: np.ndarray) -> list[FaceDetection]:
    detector = _get_mtcnn_detector()
    if detector is None:
        return []
    results = detector.detect_faces(image)
    faces: list[FaceDetection] = []
    for result in results:
        x, y, width, height = result["box"]
        confidence = float(result.get("confidence", 0.0))
        if confidence < MTCNN_MIN_CONFIDENCE:
            continue
        faces.append(
            FaceDetection(
                box=FaceBox(max(0, int(x)), max(0, int(y)), int(width), int(height)),
                confidence=confidence,
                keypoints=_extract_mtcnn_keypoints(result),
                source="mtcnn",
            )
        )
    return faces


def detect_faces_haar(image: np.ndarray) -> list[FaceDetection]:
    if cv2 is None or CASCADE is None:
        return []
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detections = CASCADE.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    return [
        FaceDetection(
            box=FaceBox(int(x), int(y), int(w), int(h)),
            confidence=0.45,
            keypoints=None,
            source="haar",
        )
        for x, y, w, h in detections
    ]


def detect_face(image: np.ndarray) -> np.ndarray | None:
    result = detect_face_result(image)
    if result is None:
        return None
    return result.face


def detect_face_result(image: np.ndarray) -> DetectedFace | None:
    detection_base, _, _ = _resize_for_detection(image)
    best_candidate: FaceCandidate | None = None
    for rotation in (0, 90, 270, 180):
        detection_image = _rotate_image(detection_base, rotation)
        detections = _detect_face_detections(detection_image)
        if not detections:
            continue
        detection = _select_best_detection(detections)
        scale_x, scale_y = _scale_factors_for_rotation(
            original_shape=image.shape,
            detection_shape=detection_image.shape,
            rotation=rotation,
        )
        detection = _scale_detection(detection, scale_x=scale_x, scale_y=scale_y)
        face = _expand_face_box(
            detection.box,
            image_shape=_rotated_image_shape(image.shape, rotation),
            margin_ratio=PRIMARY_CROP_MARGIN_RATIO,
        )
        candidate = FaceCandidate(
            detection=FaceDetection(
                box=face,
                confidence=detection.confidence,
                keypoints=detection.keypoints,
                source=detection.source,
            ),
            area=face.width * face.height,
            rotation=rotation,
        )
        if _prefer_face_candidate(candidate, best_candidate):
            best_candidate = candidate
    if best_candidate is None:
        return None
    rotated_image = _rotate_image(image, best_candidate.rotation)
    cropped = crop_face(rotated_image, best_candidate.detection.box)
    crop_keypoints = _translate_keypoints_to_crop(
        best_candidate.detection.keypoints,
        best_candidate.detection.box,
    )
    upright_crop, upright_keypoints = _normalize_face_orientation_with_keypoints(
        cropped,
        crop_keypoints,
    )
    aligned_crop = _align_face_with_keypoints(upright_crop, upright_keypoints)
    refined_crop = _refine_face_crop(aligned_crop)
    normalized_refined = _normalize_face_orientation(refined_crop)
    final_crop = _align_face(normalized_refined)
    original_box = _clip_face_box(
        _map_box_from_rotated_to_original(
            best_candidate.detection.box,
            original_shape=image.shape,
            rotation=best_candidate.rotation,
        ),
        image_shape=image.shape,
    )
    return DetectedFace(
        face=_normalize_face_orientation(final_crop),
        box=original_box,
        confidence=best_candidate.detection.confidence,
        source=best_candidate.detection.source,
        rotation=best_candidate.rotation,
    )


def crop_face(image: np.ndarray, face: FaceBox) -> np.ndarray:
    max_y, max_x = image.shape[:2]
    x_end = min(max_x, face.x + face.width)
    y_end = min(max_y, face.y + face.height)
    return image[face.y:y_end, face.x:x_end]


def draw_faces(image: np.ndarray) -> np.ndarray:
    annotated = image.copy()
    if cv2 is None:
        return annotated
    for face in detect_faces(image):
        cv2.rectangle(
            annotated,
            (face.x, face.y),
            (face.x + face.width, face.y + face.height),
            (0, 255, 0),
            2,
        )
    return annotated


def draw_face_box(
    image: np.ndarray,
    face_box: FaceBox | None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    label: str | None = None,
) -> np.ndarray:
    annotated = image.copy()
    if cv2 is None or face_box is None:
        return annotated

    x1 = max(0, int(face_box.x))
    y1 = max(0, int(face_box.y))
    x2 = max(x1 + 1, int(face_box.x + face_box.width))
    y2 = max(y1 + 1, int(face_box.y + face_box.height))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_top = max(0, y1 - text_height - baseline - 8)
        label_bottom = min(annotated.shape[0], label_top + text_height + baseline + 8)
        label_right = min(annotated.shape[1], x1 + text_width + 10)
        cv2.rectangle(annotated, (x1, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, max(text_height + 1, label_bottom - baseline - 4)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def _align_face(image: np.ndarray) -> np.ndarray:
    if cv2 is None or image.size == 0:
        return image

    angle = _estimate_eye_angle(image)
    if angle is None or abs(angle) < 2.0 or abs(angle) > 45.0:
        return image

    return _rotate_in_plane(image, angle)


def _align_face_with_keypoints(
    image: np.ndarray,
    keypoints: dict[str, tuple[float, float]] | None,
) -> np.ndarray:
    if not keypoints:
        return _align_face(image)

    angle = _estimate_eye_angle_from_keypoints(keypoints)
    if angle is None or abs(angle) < 2.0 or abs(angle) > 45.0:
        return image
    return _rotate_in_plane(image, angle)


def _estimate_eye_angle(image: np.ndarray) -> float | None:
    eye_pair = _detect_eye_pair(image)
    if eye_pair is None:
        return None

    left_eye, right_eye = eye_pair
    return float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))


def _estimate_eye_angle_from_keypoints(
    keypoints: dict[str, tuple[float, float]] | None,
) -> float | None:
    if not keypoints:
        return None

    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if left_eye is None or right_eye is None:
        return None
    left_eye, right_eye = _order_eye_pair(left_eye, right_eye)
    return float(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))


def _normalize_face_orientation(image: np.ndarray) -> np.ndarray:
    normalized, _ = _normalize_face_orientation_with_keypoints(image, None)
    return normalized


def _normalize_face_orientation_with_keypoints(
    image: np.ndarray,
    keypoints: dict[str, tuple[float, float]] | None,
) -> tuple[np.ndarray, dict[str, tuple[float, float]] | None]:
    best_image = image
    best_keypoints = keypoints
    best_score: float | None = None

    for rotation in (0, 90, 270, 180):
        candidate_image = _rotate_image(image, rotation)
        candidate_keypoints = _rotate_keypoints(keypoints, image.shape, rotation)
        score = _orientation_score(candidate_image, candidate_keypoints)
        if score is None:
            continue
        if best_score is None or score > best_score + 1e-6:
            best_image = candidate_image
            best_keypoints = candidate_keypoints
            best_score = score

    return best_image, best_keypoints


def _orientation_score(
    image: np.ndarray,
    keypoints: dict[str, tuple[float, float]] | None,
) -> float | None:
    landmarks = keypoints
    if landmarks is None:
        detections = _detect_face_detections(image)
        if detections:
            landmarks = _select_best_detection(detections).keypoints

    if landmarks:
        score = _landmark_orientation_score(image.shape, landmarks)
        if score is not None:
            return score
    return _upright_orientation_score(image)


def _upright_orientation_score(image: np.ndarray) -> float | None:
    eye_pair = _detect_eye_pair(image)
    if eye_pair is None:
        return None

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return None

    left_eye, right_eye = eye_pair
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    eye_span = max(0.0, right_eye[0] - left_eye[0])
    vertical_offset = abs(right_eye[1] - left_eye[1])
    return float(
        (eye_span / width)
        + (1.0 - (eye_mid_y / height))
        - (vertical_offset / height)
    )


def _landmark_orientation_score(
    image_shape: tuple[int, ...],
    keypoints: dict[str, tuple[float, float]] | None,
) -> float | None:
    if not keypoints:
        return None

    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if left_eye is None or right_eye is None:
        return None

    left_eye, right_eye = _order_eye_pair(left_eye, right_eye)
    height, width = image_shape[:2]
    if height == 0 or width == 0:
        return None

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    eye_span = max(0.0, right_eye[0] - left_eye[0])
    vertical_offset = abs(right_eye[1] - left_eye[1])
    score = (
        (eye_span / width)
        + (1.0 - (eye_mid_y / height))
        - (vertical_offset / height)
    )

    nose = keypoints.get("nose")
    if nose is not None:
        score += 0.30 if nose[1] > eye_mid_y else -0.30
        score += max(0.0, 1.0 - (abs(nose[0] - eye_mid_x) / max(width * 0.18, 1.0))) * 0.15

    mouth_left = keypoints.get("mouth_left")
    mouth_right = keypoints.get("mouth_right")
    if mouth_left is not None and mouth_right is not None:
        mouth_mid_y = (mouth_left[1] + mouth_right[1]) / 2.0
        score += 0.35 if mouth_mid_y > eye_mid_y else -0.35
        if nose is not None:
            score += 0.15 if mouth_mid_y > nose[1] else -0.15

    return float(score)


def _detect_eye_pair(image: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    eye_pair = _detect_eye_pair_mtcnn(image)
    if eye_pair is not None:
        return eye_pair
    return _detect_eye_pair_haar(image)


def _detect_eye_pair_mtcnn(image: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    detector = _get_mtcnn_detector()
    if detector is None:
        return None

    results = detector.detect_faces(image)
    if not results:
        return None

    best_result = max(results, key=lambda item: max(0, int(item["box"][2])) * max(0, int(item["box"][3])))
    keypoints = best_result.get("keypoints") or {}
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if left_eye is None or right_eye is None:
        return None

    return _order_eye_pair(
        (float(left_eye[0]), float(left_eye[1])),
        (float(right_eye[0]), float(right_eye[1])),
    )


def _detect_eye_pair_haar(image: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if cv2 is None or EYE_CASCADE is None:
        return None

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detections = EYE_CASCADE.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(12, 12),
    )

    height, width = grayscale.shape[:2]
    upper_limit = height * 0.7
    eyes = [
        (float(x + w / 2.0), float(y + h / 2.0))
        for x, y, w, h in detections
        if (y + h / 2.0) <= upper_limit
    ]
    return _select_best_eye_pair(eyes, width=width, height=height)


def _select_best_eye_pair(
    eyes: list[tuple[float, float]],
    *,
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    best_pair = None
    best_score = float("-inf")

    for first_index in range(len(eyes)):
        for second_index in range(first_index + 1, len(eyes)):
            left_eye, right_eye = _order_eye_pair(eyes[first_index], eyes[second_index])
            dx = right_eye[0] - left_eye[0]
            dy = abs(right_eye[1] - left_eye[1])

            if dx <= width * 0.15 or dy >= height * 0.35:
                continue

            score = dx - (dy * 2.0)
            if score > best_score:
                best_score = score
                best_pair = (left_eye, right_eye)

    return best_pair


def _prefer_face_candidate(candidate: FaceCandidate, current: FaceCandidate | None) -> bool:
    if current is None:
        return True

    if candidate.area > int(current.area * 1.08):
        return True
    if current.area > int(candidate.area * 1.08):
        return False

    candidate_rank = _rotation_priority(candidate.rotation)
    current_rank = _rotation_priority(current.rotation)
    candidate_quality = _detection_quality_score(candidate.detection) - (candidate_rank * 0.02)
    current_quality = _detection_quality_score(current.detection) - (current_rank * 0.02)
    if abs(candidate_quality - current_quality) > 0.03:
        return candidate_quality > current_quality
    if candidate_rank != current_rank:
        return candidate_rank < current_rank
    return candidate.area > current.area


def _rotation_priority(rotation: int) -> int:
    priorities = {0: 0, 90: 1, 270: 2, 180: 3}
    return priorities.get(rotation, 99)


def _order_eye_pair(
    first_eye: tuple[float, float],
    second_eye: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if first_eye[0] <= second_eye[0]:
        return first_eye, second_eye
    return second_eye, first_eye


def _translate_keypoints_to_crop(
    keypoints: dict[str, tuple[float, float]] | None,
    face: FaceBox,
) -> dict[str, tuple[float, float]] | None:
    if not keypoints:
        return None

    return {
        name: (point[0] - face.x, point[1] - face.y)
        for name, point in keypoints.items()
    }


def _rotate_keypoints(
    keypoints: dict[str, tuple[float, float]] | None,
    image_shape: tuple[int, ...],
    rotation: int,
) -> dict[str, tuple[float, float]] | None:
    if not keypoints:
        return None

    height, width = image_shape[:2]
    rotated: dict[str, tuple[float, float]] = {}
    for name, point in keypoints.items():
        x, y = point
        if rotation == 90:
            rotated[name] = (y, (width - 1) - x)
        elif rotation == 180:
            rotated[name] = ((width - 1) - x, (height - 1) - y)
        elif rotation == 270:
            rotated[name] = ((height - 1) - y, x)
        else:
            rotated[name] = (x, y)
    return rotated


def _rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return np.rot90(image, k=1).copy()
    if rotation == 180:
        return np.rot90(image, k=2).copy()
    if rotation == 270:
        return np.rot90(image, k=3).copy()
    return image


def _rotate_in_plane(image: np.ndarray, angle: float) -> np.ndarray:
    if cv2 is None:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _resize_for_detection(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    if cv2 is None:
        return image, 1.0, 1.0

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= DETECTION_MAX_DIM:
        return image, 1.0, 1.0

    scale = DETECTION_MAX_DIM / float(longest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    return resized, width / float(resized_width), height / float(resized_height)


def _scale_face_box(face: FaceBox, *, scale_x: float, scale_y: float) -> FaceBox:
    return FaceBox(
        x=max(0, int(round(face.x * scale_x))),
        y=max(0, int(round(face.y * scale_y))),
        width=max(1, int(round(face.width * scale_x))),
        height=max(1, int(round(face.height * scale_y))),
    )


def _scale_detection(
    detection: FaceDetection,
    *,
    scale_x: float,
    scale_y: float,
) -> FaceDetection:
    keypoints = None
    if detection.keypoints:
        keypoints = {
            name: (point[0] * scale_x, point[1] * scale_y)
            for name, point in detection.keypoints.items()
        }
    return FaceDetection(
        box=_scale_face_box(detection.box, scale_x=scale_x, scale_y=scale_y),
        confidence=detection.confidence,
        keypoints=keypoints,
        source=detection.source,
    )


def _scale_factors_for_rotation(
    *,
    original_shape: tuple[int, ...],
    detection_shape: tuple[int, ...],
    rotation: int,
) -> tuple[float, float]:
    rotated_width, rotated_height = _rotated_dimensions(original_shape, rotation)
    detection_height, detection_width = detection_shape[:2]
    return (
        rotated_width / float(max(detection_width, 1)),
        rotated_height / float(max(detection_height, 1)),
    )


def _rotated_dimensions(shape: tuple[int, ...], rotation: int) -> tuple[int, int]:
    height, width = shape[:2]
    if rotation in (90, 270):
        return height, width
    return width, height


def _rotated_image_shape(shape: tuple[int, ...], rotation: int) -> tuple[int, int, int]:
    height, width = shape[:2]
    channels = shape[2] if len(shape) > 2 else 1
    if rotation in (90, 270):
        return width, height, channels
    return height, width, channels


def _select_best_detection(detections: list[FaceDetection]) -> FaceDetection:
    best_detection = detections[0]
    best_area = best_detection.box.width * best_detection.box.height
    for detection in detections[1:]:
        area = detection.box.width * detection.box.height
        if area > int(best_area * 1.08):
            best_detection = detection
            best_area = area
            continue
        if best_area > int(area * 1.08):
            continue
        if _detection_quality_score(detection) > _detection_quality_score(best_detection):
            best_detection = detection
            best_area = area
    return best_detection


def _detection_quality_score(detection: FaceDetection) -> float:
    return (
        detection.confidence
        + _landmark_quality_score(detection)
        + _source_quality_bonus(detection.source)
    )


def _landmark_quality_score(detection: FaceDetection) -> float:
    keypoints = detection.keypoints or {}
    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")
    if left_eye is None or right_eye is None:
        return 0.0

    width = max(float(detection.box.width), 1.0)
    height = max(float(detection.box.height), 1.0)
    eye_span = max(0.0, right_eye[0] - left_eye[0]) / width
    eye_level_score = max(0.0, 1.0 - (abs(right_eye[1] - left_eye[1]) / (height * 0.25)))
    score = np.clip((eye_span - 0.15) / 0.35, 0.0, 1.0) * 0.35
    score += eye_level_score * 0.25

    nose = keypoints.get("nose")
    if nose is not None:
        eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
        symmetry = max(0.0, 1.0 - (abs(nose[0] - eye_mid_x) / (width * 0.2)))
        score += symmetry * 0.2

    mouth_left = keypoints.get("mouth_left")
    mouth_right = keypoints.get("mouth_right")
    if mouth_left is not None and mouth_right is not None:
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (mouth_left[1] + mouth_right[1]) / 2.0
        if mouth_mid_y > eye_mid_y:
            score += 0.2

    return float(score)


def _source_quality_bonus(source: str) -> float:
    if source == "mtcnn":
        return 0.08
    return 0.0


def _expand_face_box(
    face: FaceBox,
    *,
    image_shape: tuple[int, ...],
    margin_ratio: float,
) -> FaceBox:
    height, width = image_shape[:2]
    margin_x = int(round(face.width * margin_ratio))
    margin_y = int(round(face.height * margin_ratio))
    x = max(0, face.x - margin_x)
    y = max(0, face.y - margin_y)
    x_end = min(width, face.x + face.width + margin_x)
    y_end = min(height, face.y + face.height + margin_y)
    return FaceBox(
        x=x,
        y=y,
        width=max(1, x_end - x),
        height=max(1, y_end - y),
    )


def _refine_face_crop(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return image

    detection_image, scale_x, scale_y = _resize_for_detection(image)
    detections = _detect_face_detections(detection_image)
    if not detections:
        return image

    detection = _scale_detection(
        _select_best_detection(detections),
        scale_x=scale_x,
        scale_y=scale_y,
    )
    refined_box = _expand_face_box(
        detection.box,
        image_shape=image.shape,
        margin_ratio=REFINEMENT_CROP_MARGIN_RATIO,
    )
    refined = crop_face(image, refined_box)
    if refined.size == 0:
        return image
    return refined


def _map_box_from_rotated_to_original(
    face: FaceBox,
    *,
    original_shape: tuple[int, ...],
    rotation: int,
) -> FaceBox:
    height, width = original_shape[:2]
    if rotation == 90:
        return FaceBox(
            x=max(0, width - (face.y + face.height)),
            y=max(0, face.x),
            width=max(1, face.height),
            height=max(1, face.width),
        )
    if rotation == 180:
        return FaceBox(
            x=max(0, width - (face.x + face.width)),
            y=max(0, height - (face.y + face.height)),
            width=max(1, face.width),
            height=max(1, face.height),
        )
    if rotation == 270:
        return FaceBox(
            x=max(0, face.y),
            y=max(0, height - (face.x + face.width)),
            width=max(1, face.height),
            height=max(1, face.width),
        )
    return face


def _clip_face_box(face: FaceBox, *, image_shape: tuple[int, ...]) -> FaceBox:
    height, width = image_shape[:2]
    x = max(0, min(width - 1, face.x))
    y = max(0, min(height - 1, face.y))
    x_end = max(x + 1, min(width, face.x + face.width))
    y_end = max(y + 1, min(height, face.y + face.height))
    return FaceBox(
        x=x,
        y=y,
        width=max(1, x_end - x),
        height=max(1, y_end - y),
    )


def _extract_mtcnn_keypoints(result: dict) -> dict[str, tuple[float, float]] | None:
    raw_keypoints = result.get("keypoints") or {}
    if not raw_keypoints:
        return None
    return {
        str(name): (float(point[0]), float(point[1]))
        for name, point in raw_keypoints.items()
        if point is not None and len(point) >= 2
    } or None


def _get_mtcnn_detector():
    global _MTCNN_DETECTOR
    global _MTCNN_IMPORT_ATTEMPTED

    if _MTCNN_IMPORT_ATTEMPTED:
        return _MTCNN_DETECTOR

    _MTCNN_IMPORT_ATTEMPTED = True
    try:
        from mtcnn import MTCNN  # type: ignore
    except Exception:  # pragma: no cover
        _MTCNN_DETECTOR = None
        return None

    try:
        _MTCNN_DETECTOR = MTCNN()
    except Exception:  # pragma: no cover
        _MTCNN_DETECTOR = None
    return _MTCNN_DETECTOR
