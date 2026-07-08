# DroTrack — vendored, TensorFlow-free adaptation

This directory contains a vendored and modified copy of **DroTrack**.

- **Original author:** Ali Hamdi (ali.ali@rmit.edu.au), CRUISE Research Group, RMIT.
- **Original project:** https://github.com/cruiseresearchgroup/DroTrack
- **Paper:** "DroTrack: High-speed Drone-based Object Tracking Under Uncertainty",
  IEEE FUZZ 2020. https://arxiv.org/abs/2005.00828
- **License:** MIT (retained; see the upstream repository).

## Changes made in this copy

1. **Removed the TensorFlow / Keras dependency.** The original used VGG16
   (ImageNet) deep features only to rank fuzzy-segmentation candidates by
   similarity to the tracked template. That is replaced by a lightweight
   fixed-length appearance descriptor in [`features.py`](features.py). This
   makes the tracker dependency-light and fast on a laptop CPU, at some cost to
   the fidelity of the original segment-matching step. **This copy is therefore
   a derivative, not a bit-exact reproduction.**
2. **Packaging:** absolute imports (`utils.*`, `models.*`) rewritten as relative
   imports; `run.py`, dataset loaders, and result-saving utilities were not
   vendored (not needed for live tracking).
3. **Robustness:** the optical-flow call now coerces the previous-points array
   to the `(N,1,2)` float32 shape OpenCV expects.

The tracking algorithm itself — Lucas-Kanade optical flow, angular relative
scaling, and fuzzy c-means segmentation — is unchanged.
