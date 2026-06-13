from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from database.db_operations import _build_unknown_face_destination
from src.pipeline import (
    _filter_registered_training_images,
    _drop_conflicting_duplicate_images,
    _is_legacy_rotation_variant,
    _is_preprocessed_face_image,
    exemplar_verify,
    collect_dataset,
    parse_label_from_path,
    resolve_prediction,
)


def _write_image(path: Path, value: int) -> None:
    image = np.full((160, 160, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_parse_label_from_path_supports_flat_file_names():
    label = parse_label_from_path(Path("dataset/train/p12_3.jpg"))
    assert label == 12


def test_collect_dataset_reads_images():
    train_dir = Path("D:/Criminal_Identification_System/tests/_tmp_train")
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir()
    try:
        (train_dir / "criminal_1").mkdir()
        (train_dir / "criminal_2").mkdir()
        _write_image(train_dir / "criminal_1" / "p1_1_orig.png", 40)
        _write_image(train_dir / "criminal_2" / "p2_1_orig.png", 180)

        X, y, paths = collect_dataset(train_dir)

        assert X.shape[0] == 2
        assert X.shape[1] == 512
        assert sorted(y.tolist()) == [1, 2]
        assert len(paths) == 2
    finally:
        if train_dir.exists():
            shutil.rmtree(train_dir)


def test_legacy_rotation_variants_are_skipped() -> None:
    assert _is_legacy_rotation_variant(Path("dataset/train/criminal_4/p4_1_rot90.png"))
    assert not _is_legacy_rotation_variant(Path("dataset/train/criminal_5/p5_1_orig.png"))


def test_preprocessed_face_images_are_detected() -> None:
    assert _is_preprocessed_face_image(Path("dataset/train/criminal_5/p5_1_orig.png"))
    assert not _is_preprocessed_face_image(Path("dataset/train/criminal_1/p1_1.jpg"))


def test_conflicting_duplicates_are_removed():
    train_dir = Path("D:/Criminal_Identification_System/tests/_tmp_dupes")
    if train_dir.exists():
        shutil.rmtree(train_dir)
    (train_dir / "criminal_1").mkdir(parents=True)
    (train_dir / "criminal_2").mkdir(parents=True)
    try:
        image = np.full((40, 40, 3), 120, dtype=np.uint8)
        Image.fromarray(image).save(train_dir / "criminal_1" / "p1_1.jpg")
        Image.fromarray(image).save(train_dir / "criminal_2" / "p2_1.jpg")
        kept = _drop_conflicting_duplicate_images(
            [
                train_dir / "criminal_1" / "p1_1.jpg",
                train_dir / "criminal_2" / "p2_1.jpg",
            ]
        )
        assert kept == []
    finally:
        if train_dir.exists():
            shutil.rmtree(train_dir)


def test_filter_registered_training_images_excludes_orphans():
    image_paths = [
        Path("D:/Criminal_Identification_System/dataset/train/criminal_1/p1_1_orig.png"),
        Path("D:/Criminal_Identification_System/dataset/train/criminal_2/p2_1_orig.png"),
    ]
    registered = {image_paths[0]}

    kept = _filter_registered_training_images(image_paths, registered)

    assert kept == [image_paths[0]]


def test_exemplar_verify_accepts_clear_best_match():
    exemplar_model = {
        "embeddings": np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [10.0, 10.0],
            ],
            dtype=np.float32,
        ),
        "labels": np.array([5, 5, 4], dtype=int),
    }

    label, confidence, known = exemplar_verify(np.array([0.2, 0.1], dtype=np.float32), exemplar_model)

    assert label == 5
    assert known is True
    assert confidence > 0.0


def test_resolve_prediction_prefers_matching_exemplar():
    label, confidence, unknown = resolve_prediction(
        mlp_label=5,
        mlp_confidence=0.99,
        exemplar_label=5,
        exemplar_confidence=0.7,
        exemplar_known=True,
        transformed_selected=np.array([0.0, 0.0], dtype=np.float32),
        centroid_model={"centroids": {5: np.array([0.0, 0.0], dtype=np.float32)}, "thresholds": {5: 1.0}},
    )

    assert label == 5
    assert unknown is False
    assert confidence == 0.99


def test_resolve_prediction_rejects_weak_echoed_support():
    label, confidence, unknown = resolve_prediction(
        mlp_label=5,
        mlp_confidence=0.86,
        exemplar_label=5,
        exemplar_confidence=0.04,
        exemplar_known=True,
        transformed_selected=np.array([1.0, 0.0], dtype=np.float32),
        centroid_model={"centroids": {5: np.array([0.0, 1.0], dtype=np.float32)}, "thresholds": {5: 2.0}},
    )

    assert label is None
    assert unknown is True
    assert confidence >= 0.86


def test_unknown_face_destination_keeps_same_file_path():
    output_dir = Path("D:/Criminal_Identification_System/tests/_tmp_unknown_faces")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    try:
        source = output_dir / "face.jpg"
        source.write_bytes(b"demo")
        destination = _build_unknown_face_destination(output_dir, source)
        assert destination == source
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
