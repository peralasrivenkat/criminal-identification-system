# Project Documentation

## Project Title

**Criminal Identification System**

Major project by **Perala Srivenkat** as **Team Lead**.

GitHub profile: [peralasrivenkat](https://github.com/peralasrivenkat)

## Project Summary

This project is a desktop-based criminal identification system that uses face recognition and machine learning to identify registered criminals from uploaded images, videos, live camera input, or CCTV streams.

The system supports two major workflows:

1. **Training / Registration Workflow**
   Register criminal details, upload source face images, detect and align faces, preprocess the images, generate training samples, and train the recognition pipeline.

2. **Testing / Identification Workflow**
   Upload an unknown image or video, detect and align the face, extract embeddings, apply feature optimization, classify the person, match the predicted ID with the database, and show criminal details in the GUI.

## Team Lead Contribution

As Team Lead, I handled:

- Overall project planning and module division.
- GUI workflow design for Admin Panel and Identification App.
- Face preprocessing, alignment, and dataset preparation flow.
- Machine learning pipeline integration.
- PCA, ACO, MLP model workflow coordination.
- MySQL database integration.
- Debugging wrong predictions, rotated face crops, and model artifact issues.
- GitHub documentation and project presentation preparation.

## Overall Methodology

```text
Image / Video / Camera / CCTV Input
        |
        v
Face Detection using MTCNN / Haar Cascade
        |
        v
Face Alignment using eye landmarks
        |
        v
Preprocessing: RGB conversion, denoising, resize 160 x 160, normalization
        |
        v
512-D Face Embedding Extraction
        |
        v
Feature Scaling using StandardScaler
        |
        v
PCA Dimensionality Reduction
        |
        v
ACO Feature Selection
        |
        v
MLP Classification
        |
        v
Database Matching
        |
        v
GUI Result Display and Prediction Logging
```

## Technology Stack

| Area | Technology |
| --- | --- |
| Programming Language | Python |
| GUI Framework | Tkinter |
| Image Processing | OpenCV, Pillow |
| Face Detection | MTCNN, Haar Cascade |
| Face Embedding | FaceNet-compatible runtime or fallback 512-D descriptor |
| Machine Learning | scikit-learn |
| Scaling | StandardScaler |
| Dimensionality Reduction | PCA |
| Feature Selection | Ant Colony Optimization |
| Classification | MLPClassifier |
| Database | MySQL |
| Model Storage | joblib, JSON |
| Testing | pytest |
| Version Control | Git, GitHub |

## Installed Libraries

The required libraries are listed in `requirements.txt`:

```text
numpy
opencv-python
scikit-learn
joblib
Pillow
pytest
mysql-connector-python
```

Optional libraries:

```text
mtcnn
tensorflow
```

These optional libraries are used when the advanced MTCNN detector and real FaceNet model runtime are available.

## Folder And File Details

### Root Files

| File | Purpose |
| --- | --- |
| `README.md` | Main GitHub landing page with project summary, screenshots, setup, methodology, and usage. |
| `PROJECT_DOCUMENTATION.md` | Full detailed explanation of project files, workflow, algorithms, and architecture. |
| `config.py` | Central configuration for paths, image size, embedding size, PCA/ACO settings, thresholds, and directory creation. |
| `main.py` | Command-line entry point for launching GUI, initializing DB, training, image prediction, and video prediction. |
| `requirements.txt` | Python dependency list. |
| `pytest.ini` | pytest configuration. |
| `.gitignore` | Prevents private datasets, trained models, logs, outputs, caches, and databases from being uploaded. |

### `database/`

This folder manages MySQL database setup, connections, criminal records, image paths, and prediction logs.

| File | Purpose |
| --- | --- |
| `database/__init__.py` | Marks the folder as a Python package. |
| `database/db_config.py` | Stores MySQL connection details such as host, user, password, database, and port. |
| `database/db_connection.py` | Creates MySQL server and database connections. |
| `database/db_operations.py` | Contains database operations for adding criminals, storing image paths, fetching records, registering criminals with images, storing unknown faces, and logging predictions. |
| `database/schema.sql` | SQL schema for `criminals`, `criminal_images`, and `prediction_logs` tables. |

### `gui/`

This folder contains the Tkinter desktop application.

| File | Purpose |
| --- | --- |
| `gui/__init__.py` | Marks the GUI folder as a Python package. |
| `gui/home.py` | Main dashboard screen. Shows registered criminals, training sample count, embedding backend, and navigation buttons. |
| `gui/admin_panel.py` | Admin module for registering criminal profiles and uploading source images. |
| `gui/app.py` | Identification module for image, video, camera, and CCTV recognition. |
| `gui/ui_theme.py` | Shared GUI styling, buttons, panels, preview rendering, scrollable layout, and table styling. |
| `gui/assets/back.png` | Back navigation icon. |
| `gui/assets/logo.png` | Header logo used in the GUI. |
| `gui/assets/next.png` | Next image navigation icon. |
| `gui/assets/previous.png` | Previous image navigation icon. |

### `src/`

This folder contains the machine learning and face recognition pipeline.

| File | Purpose |
| --- | --- |
| `src/__init__.py` | Marks the source folder as a Python package. |
| `src/preprocessing.py` | Loads images, corrects EXIF orientation, converts to RGB, denoises, enhances contrast, resizes to 160 x 160, and normalizes pixel values. |
| `src/face_detection.py` | Detects faces using MTCNN/Haar, selects the best face, corrects orientation, aligns eyes horizontally, refines crops, and draws face boxes. |
| `src/facenet_runtime.py` | Loads a real TensorFlow FaceNet `.pb` or checkpoint model when available and extracts normalized embeddings. |
| `src/embedding.py` | Main embedding interface. Uses FaceNet when available, otherwise generates a fallback 512-D descriptor. |
| `src/pca_module.py` | Trains and applies StandardScaler and PCA. PCA reduces embedding features to a smaller optimized space. |
| `src/aco_module.py` | Implements Ant Colony Optimization for selecting the best PCA features. |
| `src/classifier.py` | Trains the MLP classifier and supports single-class fallback behavior. |
| `src/pipeline.py` | Main orchestration file for dataset collection, training, artifact loading, prediction, video recognition, registration pipeline, centroid verification, and exemplar verification. |
| `src/train_classifier.py` | Older helper script for classifier training experiments. |

### `models/`

This folder contains model training helpers. Runtime `.pkl` files are ignored by GitHub because they are generated artifacts.

| File | Purpose |
| --- | --- |
| `models/train_models.py` | Helper script that runs `train_pipeline()` and prints training metrics. |

Generated model files are intentionally ignored:

```text
scaler.pkl
pca_model.pkl
aco_features.pkl
mlp_model.pkl
centroid_model.pkl
exemplar_model.pkl
training_metadata.json
facenet_model.pb
models/facenet/
```

These files are regenerated during training and may contain dataset-specific information.

### `scripts/`

| File | Purpose |
| --- | --- |
| `scripts/repair_face_crops.py` | Repairs saved face crop images by re-running face detection, upright orientation correction, and eye alignment. |

### `tests/`

This folder contains the actual test suite.

| File | Purpose |
| --- | --- |
| `tests/test_face_preparation.py` | Tests face preprocessing, flip-only augmentation, detection alignment, orientation correction, and detection resizing. |
| `tests/test_pipeline.py` | Tests label parsing, dataset collection, duplicate filtering, registered-image filtering, exemplar verification, and unknown-person logic. |
| `tests/test_pca.py` | Tests PCA dimensionality reduction. |
| `tests/test_aco.py` | Tests that ACO returns valid selected feature indices. |
| `tests/test_classifier_training.py` | Tests MLP probability output and single-class fallback behavior. |
| `tests/test_facenet_runtime.py` | Tests FaceNet prewhitening and missing-model status handling. |

### `utils/`

| File | Purpose |
| --- | --- |
| `utils/helpers.py` | Utility helper module reserved for common helper functions. |
| `utils/metrics.py` | Utility metrics module reserved for evaluation helpers. |

### `docs/images/`

This folder stores screenshots used in GitHub documentation.

| File | Description |
| --- | --- |
| `docs/images/01-dashboard.png` | Main dashboard showing registered criminals, training samples, embedding backend, and navigation. |
| `docs/images/02-admin-panel.png` | Empty Admin Panel before criminal registration. |
| `docs/images/03-registration-details.png` | Criminal details form filled with uploaded images. |
| `docs/images/04-registered-records.png` | Registered Criminals table showing database records. |
| `docs/images/05-identification-app.png` | Identification App screen for image/video/live recognition. |
| `docs/images/06-uploaded-image.png` | Uploaded suspect image shown in preview. |
| `docs/images/07-prediction-result.png` | Final prediction result showing ID, confidence, and criminal details. |

## Training Workflow In Detail

1. The admin opens `gui/admin_panel.py`.
2. The admin enters criminal details.
3. The admin uploads exactly 5 source images.
4. `database/db_operations.py` receives the details and image paths.
5. Each image is loaded using `src/preprocessing.py`.
6. A face is detected using `src/face_detection.py`.
7. The face is aligned using eye landmarks.
8. The face is preprocessed to `160 x 160` RGB format.
9. Two training variants are saved: original and horizontal flip.
10. Image paths are stored in MySQL.
11. Training reads each saved face image as an independent sample.
12. `src/embedding.py` extracts a 512-D embedding.
13. `src/pca_module.py` scales features and applies PCA.
14. `src/aco_module.py` selects the best features.
15. `src/classifier.py` trains the MLP classifier.
16. `src/pipeline.py` saves scaler, PCA, ACO, MLP, centroid, exemplar, and metadata artifacts.

## Identification Workflow In Detail

1. The user opens the Identification App.
2. The user uploads an image, video, or starts camera/CCTV recognition.
3. The input frame is loaded and normalized.
4. The face is detected and aligned.
5. The face is resized to `160 x 160` RGB.
6. A 512-D embedding is extracted.
7. The trained scaler normalizes the embedding.
8. PCA transforms it into reduced feature space.
9. ACO-selected features are kept.
10. MLP predicts the most likely criminal ID.
11. Exemplar and centroid verification check if the match is strong enough.
12. If accepted, the predicted ID is matched with MySQL criminal records.
13. If rejected, the result is shown as `Unknown Person`.
14. The prediction event is logged in the database.
15. Unknown images are stored in `outputs/unknown_faces/`.

## Algorithm Explanation

### MTCNN / Haar Cascade

MTCNN is used for face detection and landmark extraction. It helps identify the face bounding box and facial points such as eyes, nose, and mouth. Haar Cascade is used as a fallback detector when MTCNN is unavailable or does not detect a face.

### Face Alignment

Face alignment rotates the cropped face so that both eyes are horizontal and the nose is below the eye line. This improves consistency between training and testing images.

### Preprocessing

Preprocessing standardizes the face image before feature extraction. The project performs RGB conversion, noise removal, contrast enhancement, resizing to `160 x 160`, and pixel normalization.

### FaceNet / Embedding Extraction

The system supports FaceNet-compatible embedding extraction. If a pretrained FaceNet model is available, the system loads it and extracts a normalized embedding. If not, a fallback 512-D descriptor is generated using low-frequency image structure, gradients, DCT, LBP, HSV color histograms, and global statistics.

### StandardScaler

StandardScaler normalizes feature values so that each feature has a comparable scale. This improves PCA and classifier performance.

### PCA

PCA reduces dimensionality by preserving the most important variance in the embedding data. This reduces redundant information, improves speed, and can reduce overfitting.

### ACO

Ant Colony Optimization selects the best subset of PCA features. Features are treated like paths, and the algorithm updates pheromone values to prefer features that improve class separation and reduce redundancy.

### MLP Classifier

The MLP classifier is the final supervised classification model. It learns nonlinear relationships in the optimized feature space and predicts the criminal ID.

### Exemplar And Centroid Verification

The project also uses verification models:

- **Exemplar model:** compares the test embedding with stored training embeddings.
- **Centroid model:** compares the selected feature vector with the average center of each class.

These verification layers reduce wrong forced predictions and help return `Unknown Person` when confidence is not strong enough.

## Database Tables

### `criminals`

Stores criminal profile details:

```text
id
name
dob
moles
nationality
region
crime
num_crimes
created_at
```

### `criminal_images`

Stores training image paths linked to each criminal:

```text
id
criminal_id
image_path
created_at
```

### `prediction_logs`

Stores prediction history:

```text
id
source_type
source_ref
is_criminal
predicted_label
confidence
status
embedding_json
selected_features_json
created_at
```

## Files Intentionally Excluded From GitHub

The `.gitignore` excludes private and generated data:

```text
dataset/train/
dataset/test/
models/*.pkl
models/*.json
models/*.pb
models/facenet/
logs/
outputs/
results/
*.db
*.sqlite
__pycache__/
.pytest_cache/
```

These files are excluded because they can contain personal face images, trained biometric artifacts, prediction outputs, logs, or local runtime data.

## How To Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Initialize the database:

```powershell
python main.py init-db
```

Open the full GUI:

```powershell
python main.py gui
```

Train the model:

```powershell
python main.py train
```

Identify an image:

```powershell
python main.py identify path\to\image.jpg
```

Identify a video:

```powershell
python main.py identify-video path\to\video.mp4
```

Run tests:

```powershell
python -m pytest tests
```

## Resume Description

**Criminal Identification System | Team Lead**

Developed a Python-based criminal identification system using face recognition and machine learning. The system supports criminal registration, face detection, face alignment, 160 x 160 RGB preprocessing, 512-D embedding extraction, feature scaling, PCA dimensionality reduction, ACO feature selection, MLP classification, MySQL database matching, and GUI-based result display for image, video, camera, and CCTV input.

## Future Scope

- Add a stronger pretrained FaceNet or ArcFace model.
- Improve real-time multi-face tracking.
- Add admin authentication and role-based access.
- Add a web dashboard for prediction history.
- Add cloud database and deployment support.
- Improve unknown-person rejection with a larger validation dataset.
- Add model comparison reports for MLP, SVM, KNN, and CNN classifiers.
