"""Lightweight appearance descriptor -- the TensorFlow-free replacement for
DroTrack's original VGG16 (Keras) feature extractor.

DroTrack used VGG16 deep features only to pick, among the fuzzy-segmentation
candidates, the patch most similar to the tracked template (via cosine
distance). Running VGG16 needs TensorFlow and a GPU to be fast; on a laptop CPU
it dominates the runtime. We swap it for a small, fixed-length appearance vector
(downscaled intensity), which keeps the "most similar patch" selection while
staying dependency-light and fast. This makes our DroTrack a derivative of the
original (see NOTICE)."""

import cv2
import numpy as np

_SIZE = 16   # descriptor is _SIZE*_SIZE = 256-d


def features(image, model=None, preprocess_input=None):
    """Return a fixed-length L2-normalized appearance vector for an image patch.
    `model`/`preprocess_input` are accepted and ignored for drop-in
    compatibility with the original cnn.features(image, model, preprocess)."""
    if image is None or image.size == 0:
        return np.zeros(_SIZE * _SIZE, dtype=np.float32)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(image, (_SIZE, _SIZE)).astype(np.float32).flatten()
    norm = np.linalg.norm(patch)
    return patch / norm if norm > 0 else patch
