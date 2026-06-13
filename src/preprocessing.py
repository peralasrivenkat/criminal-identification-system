from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from config import IMAGE_SIZE

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


def load_image(image_path: str | Path, max_dim: int | None = None) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    with Image.open(path) as image:
        corrected = ImageOps.exif_transpose(image)
        if max_dim and max(corrected.size) > max_dim:
            resample = getattr(Image, "Resampling", Image).LANCZOS
            corrected.thumbnail((max_dim, max_dim), resample)
        return np.array(corrected.convert("RGB"))


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return np.array(Image.fromarray(image).convert("L"))


def remove_noise(image: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)
        return cv2.GaussianBlur(image, (3, 3), 0)
    return np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=1)))


def resize_image(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return np.array(Image.fromarray(image).resize(size, resample))


def normalize_image(image: np.ndarray) -> np.ndarray:
    normalized = image.astype("float32")
    return normalized / 255.0


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    return np.array(Image.fromarray(image).convert("RGB"))


def enhance_face_contrast(image: np.ndarray) -> np.ndarray:
    rgb_image = ensure_rgb(image)
    if cv2 is None:
        return np.array(ImageOps.autocontrast(Image.fromarray(rgb_image)))

    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def prepare_face_image(
    image: np.ndarray,
    size: tuple[int, int] = IMAGE_SIZE,
    *,
    denoise: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    rgb_image = ensure_rgb(image)
    processed = remove_noise(rgb_image) if denoise else rgb_image
    enhanced = enhance_face_contrast(processed)
    resized = resize_image(enhanced, size=size)
    if normalize:
        return normalize_image(resized)
    return resized.astype("uint8")


def preprocess_image(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    return prepare_face_image(image, size=size)


def preprocess_grayscale(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    grayscale = convert_to_grayscale(image)
    denoised = remove_noise(grayscale)
    resized = resize_image(denoised, size=size)
    return normalize_image(resized)


def prepare_facenet_input(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    return prepare_face_image(image, size=size)


def save_rgb_image(image: np.ndarray, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image, 0, 255).astype("uint8")
    Image.fromarray(array, mode="RGB").save(output_path)
