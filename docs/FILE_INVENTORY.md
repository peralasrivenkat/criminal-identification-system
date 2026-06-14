# Repository File Inventory

This file explains what is included in the GitHub repository and what is intentionally kept local.

## Included

- `README.md`: Main GitHub project page.
- `PROJECT_DOCUMENTATION.md`: Detailed file-by-file and algorithm documentation.
- `config.py`: Global paths and ML settings.
- `main.py`: CLI entry point.
- `requirements.txt`: Required Python libraries.
- `database/`: MySQL connection, schema, and database operations.
- `gui/`: Tkinter desktop GUI screens and assets.
- `src/`: Face recognition, preprocessing, embedding, PCA, ACO, MLP, and pipeline code.
- `models/train_models.py`: Training helper script.
- `scripts/repair_face_crops.py`: Utility for repairing saved face crops.
- `tests/`: pytest test suite.
- `docs/images/`: Prototype screenshots used in README.
- Folder README files for dataset, models, outputs, logs, and results.

## Kept Local

The following are intentionally excluded from GitHub:

- `dataset/train/` real face images.
- `dataset/test/` real test images.
- `models/*.pkl` trained model artifacts.
- `models/*.pb` and FaceNet checkpoint files.
- `outputs/unknown_faces/` uploaded unknown-person images.
- `logs/*.log` runtime logs.
- `results/*` generated accuracy and evaluation files.
- `resume_project_update/` generated resume PDFs and Word files.
- `__pycache__/`, `.pytest_cache/`, and other cache folders.

These files are excluded because they may contain private biometric data, generated model binaries, local runtime information, or files that can be regenerated.
