# Dataset Folder

This folder stores local training and testing images used by the Criminal Identification System.

The actual face image datasets are intentionally not uploaded to GitHub because they may contain private biometric data.

Expected structure:

```text
dataset/
|-- train/
|   |-- criminal_1/
|   |-- criminal_2/
|   `-- ...
`-- test/
```

During registration, the application detects and aligns each face, preprocesses it to `160 x 160` RGB format, and saves training samples inside `dataset/train/criminal_<id>/`.
