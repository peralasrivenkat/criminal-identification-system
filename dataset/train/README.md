# Training Dataset

This folder is generated locally when criminals are registered through the Admin Panel.

Each registered criminal gets a class folder:

```text
dataset/train/criminal_<id>/
```

Each uploaded source image is converted into:

```text
p<id>_<image_number>_orig.png
p<id>_<image_number>_flip.png
```

The real training images are not included in the GitHub repository because they may contain personal face data.
