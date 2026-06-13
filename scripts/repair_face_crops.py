from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf  # type: ignore

    tf.keras.utils.disable_interactive_logging()
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.face_detection import _align_face, _normalize_face_orientation, detect_face
from src.preprocessing import load_image, save_rgb_image


def repair_image(path: Path) -> None:
    image = load_image(path)
    detected = detect_face(image)
    if detected is not None:
        repaired = detected
    else:
        repaired = _normalize_face_orientation(image)
        repaired = _align_face(repaired)
        repaired = _normalize_face_orientation(repaired)
    save_rgb_image(repaired, path)


def iter_pngs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.png"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair saved face crops to upright aligned orientation.")
    parser.add_argument("root", nargs="?", default="dataset/train", help="PNG file or folder to repair")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    targets = iter_pngs(root)
    updated = 0
    errors: list[tuple[Path, str]] = []

    for target in targets:
        try:
            repair_image(target)
            updated += 1
        except Exception as error:  # pragma: no cover - operational fallback
            errors.append((target, str(error)))

    print(f"updated={updated}")
    print(f"errors={len(errors)}")
    for target, error in errors[:10]:
        print(f"{target} :: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
