from __future__ import annotations

import json
import shutil
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from config import DATASET_DIR, REGISTRATION_IMAGE_COUNT, SCHEMA_PATH, SUPPORTED_IMAGE_EXTENSIONS, ensure_directories
from database.db_config import DB_CONFIG
from database.db_connection import create_connection, create_server_connection
from src.face_detection import detect_face
from src.preprocessing import load_image, prepare_facenet_input, save_rgb_image


def initialize_database() -> None:
    ensure_directories()
    server_connection = create_server_connection()
    try:
        cursor = server_connection.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        server_connection.commit()
        cursor.close()
    finally:
        server_connection.close()

    connection = create_connection()
    try:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        cursor = connection.cursor()
        for statement in [part.strip() for part in schema.split(";") if part.strip()]:
            cursor.execute(statement)
        connection.commit()
    finally:
        connection.close()


def add_criminal(
    name: str,
    dob: str | None,
    moles: str,
    nationality: str,
    region: str,
    crime: str,
    num_crimes: int,
    criminal_id: int | None = None,
) -> int:
    initialize_database()
    connection = create_connection()
    try:
        cursor = connection.cursor()
        if criminal_id is None:
            cursor.execute(
                """
                INSERT INTO criminals (name, dob, moles, nationality, region, crime, num_crimes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (name, dob, moles, nationality, region, crime, num_crimes),
            )
        else:
            cursor.execute("DELETE FROM criminals WHERE id = %s", (criminal_id,))
            cursor.execute(
                """
                INSERT INTO criminals (
                    id, name, dob, moles, nationality, region, crime, num_crimes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (criminal_id, name, dob, moles, nationality, region, crime, num_crimes),
            )
        connection.commit()
        return int(cursor.lastrowid or criminal_id)
    finally:
        connection.close()


def add_criminal_images(criminal_id: int, image_paths: Iterable[str]) -> None:
    initialize_database()
    connection = create_connection()
    try:
        cursor = connection.cursor()
        for image_path in image_paths:
            cursor.execute(
                "INSERT INTO criminal_images (criminal_id, image_path) VALUES (%s, %s)",
                (criminal_id, str(image_path)),
            )
        connection.commit()
    finally:
        connection.close()


def replace_criminal_images(criminal_id: int, image_paths: Iterable[str]) -> None:
    initialize_database()
    connection = create_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM criminal_images WHERE criminal_id = %s", (criminal_id,))
        for image_path in image_paths:
            cursor.execute(
                "INSERT INTO criminal_images (criminal_id, image_path) VALUES (%s, %s)",
                (criminal_id, str(image_path)),
            )
        connection.commit()
    finally:
        connection.close()


def get_criminal(criminal_id: int) -> tuple[dict | None, list[dict]]:
    initialize_database()
    connection = create_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM criminals WHERE id = %s", (criminal_id,))
        criminal_row = cursor.fetchone()
        cursor.execute(
            "SELECT image_path FROM criminal_images WHERE criminal_id = %s ORDER BY id",
            (criminal_id,),
        )
        image_rows = cursor.fetchall()
        criminal = criminal_row if criminal_row else None
        images = list(image_rows)
        return criminal, images
    finally:
        connection.close()


def get_criminal_record(criminal_id: int) -> dict | None:
    initialize_database()
    connection = create_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM criminals WHERE id = %s", (criminal_id,))
        return cursor.fetchone()
    finally:
        connection.close()


def has_meaningful_criminal_details(criminal: dict | None) -> bool:
    if not criminal:
        return False

    name = (criminal.get("name") or "").strip()
    moles = (criminal.get("moles") or "").strip()
    nationality = (criminal.get("nationality") or "").strip()
    region = (criminal.get("region") or "").strip()
    crime = (criminal.get("crime") or "").strip()
    num_crimes = criminal.get("num_crimes") or 0

    if name.lower().startswith("criminal "):
        name = ""
    if nationality.lower() == "unknown":
        nationality = ""
    if region.lower() == "unknown":
        region = ""
    if crime.lower() == "unknown":
        crime = ""

    return any([name, criminal.get("dob"), moles, nationality, region, crime, int(num_crimes) > 0])


def get_all_criminals() -> list[dict]:
    initialize_database()
    connection = create_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM criminals ORDER BY id")
        return list(cursor.fetchall())
    finally:
        connection.close()


def get_registered_training_image_paths() -> set[Path]:
    try:
        initialize_database()
        connection = create_connection()
    except Exception as error:
        logging.warning("Unable to read registered training images from the database: %s", error)
        return set()

    try:
        cursor = connection.cursor()
        cursor.execute("SELECT image_path FROM criminal_images ORDER BY id")
        rows = cursor.fetchall()
        registered_paths: set[Path] = set()
        for (image_path,) in rows:
            if not image_path:
                continue
            path = Path(str(image_path))
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            registered_paths.add(path.resolve())
        return registered_paths
    except Exception as error:
        logging.warning("Unable to query criminal_images for training paths: %s", error)
        return set()
    finally:
        connection.close()


def upsert_criminal_record(
    criminal_id: int,
    name: str | None = None,
    image_paths: Iterable[str] | None = None,
) -> int:
    existing = get_criminal_record(criminal_id)
    if existing is None:
        criminal_name = name or f"Criminal {criminal_id}"
        new_id = add_criminal(
            name=criminal_name,
            dob=None,
            moles="",
            nationality="Unknown",
            region="Unknown",
            crime="Unknown",
            num_crimes=0,
            criminal_id=criminal_id,
        )
    else:
        new_id = criminal_id

    if image_paths:
        replace_criminal_images(new_id, image_paths)
    return new_id


def seed_database_from_dataset(dataset_dir: Path = DATASET_DIR) -> int:
    initialize_database()
    grouped_images: dict[int, list[str]] = {}
    for image_path in dataset_dir.rglob("*"):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        criminal_id = _extract_id_from_path(image_path)
        grouped_images.setdefault(criminal_id, []).append(str(image_path))

    inserted = 0
    for criminal_id, image_paths in sorted(grouped_images.items()):
        if criminal_id == 0:
            continue
        upsert_criminal_record(criminal_id, image_paths=image_paths)
        inserted += 1
    return inserted


def store_unknown_face(source_image: str | Path) -> Path:
    ensure_directories()
    source_path = Path(source_image)
    destination_dir = DATASET_DIR.parent / "outputs" / "unknown_faces"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = _build_unknown_face_destination(destination_dir, source_path)
    if source_path.resolve() == destination.resolve():
        return source_path
    shutil.copy2(source_path, destination)
    return destination


def log_prediction_event(
    *,
    source_type: str,
    source_ref: str | Path | None,
    is_criminal: bool,
    predicted_label: int | None,
    confidence: float,
    status: str,
    embedding: object,
    selected_features: object,
) -> None:
    initialize_database()
    connection = create_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_logs (
                source_type,
                source_ref,
                is_criminal,
                predicted_label,
                confidence,
                status,
                embedding_json,
                selected_features_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                source_type,
                str(source_ref) if source_ref is not None else None,
                1 if is_criminal else 0,
                predicted_label,
                float(confidence),
                status,
                _vector_to_json(embedding),
                _vector_to_json(selected_features),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def register_criminal_with_images(
    *,
    name: str,
    dob: str | None,
    moles: str,
    nationality: str,
    region: str,
    crime: str,
    num_crimes: int,
    image_paths: Iterable[str],
) -> tuple[int, list[str]]:
    ensure_directories()
    source_images = [Path(path) for path in image_paths]
    if len(source_images) != REGISTRATION_IMAGE_COUNT:
        raise ValueError(f"Exactly {REGISTRATION_IMAGE_COUNT} images are required to register a criminal.")

    criminal_id = add_criminal(
        name=name,
        dob=dob,
        moles=moles,
        nationality=nationality,
        region=region,
        crime=crime,
        num_crimes=num_crimes,
    )

    destination_dir = DATASET_DIR / "train" / f"criminal_{criminal_id}"
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    stored_paths: list[str] = []
    for index, source_image in enumerate(source_images, start=1):
        if source_image.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {source_image.name}")

        image = load_image(source_image)
        face = detect_face(image)
        if face is None:
            raise ValueError(f"No face detected in {source_image.name}. Please upload a clearer face image.")

        variants = _generate_face_variants(face)
        for variant_name, variant_image in variants:
            prepared = prepare_facenet_input(variant_image)
            destination_path = destination_dir / f"p{criminal_id}_{index}_{variant_name}.png"
            save_rgb_image(prepared * 255.0, destination_path)
            stored_paths.append(str(destination_path))

    replace_criminal_images(criminal_id, stored_paths)
    return criminal_id, stored_paths


def _extract_id_from_path(path: Path) -> int:
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
    return int(digits) if digits else 0


def _generate_face_variants(face: object) -> list[tuple[str, object]]:
    import numpy as np

    image = np.asarray(face)
    variants = [
        ("orig", image),
        ("flip", np.fliplr(image).copy()),
    ]
    return variants


def _vector_to_json(vector: object) -> str:
    array = np.asarray(vector, dtype="float32").reshape(-1)
    return json.dumps(array.tolist(), separators=(",", ":"))


def _build_unknown_face_destination(destination_dir: Path, source_path: Path) -> Path:
    candidate = destination_dir / source_path.name
    if not candidate.exists():
        return candidate
    if source_path.resolve() == candidate.resolve():
        return candidate

    suffix_index = 1
    while True:
        candidate = destination_dir / f"{source_path.stem}_{suffix_index}{source_path.suffix}"
        if not candidate.exists():
            return candidate
        suffix_index += 1
