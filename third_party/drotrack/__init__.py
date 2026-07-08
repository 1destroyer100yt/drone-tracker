"""TensorFlow-free DroTrack, vendored as an optional object-tracking backend.

Exposes `DroTrackCV`, a thin adapter that presents the same `init(frame, box)` /
`update(frame) -> (ok, box)` interface as OpenCV's trackers, so it drops into
ObjectFollower alongside CSRT/KCF. DroTrack works on grayscale frames; the
adapter converts internally.

See NOTICE for original authorship (Ali Hamdi, MIT) and the changes made.
"""

import cv2
import numpy as np

from .drotrack import DroTrack


class DroTrackCV:
    """Adapts DroTrack to OpenCV's tracker init/update contract."""

    def __init__(self):
        self._dt = None

    @staticmethod
    def _gray(frame):
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def init(self, frame, box):
        """box = (x, y, w, h). Returns True on success, False if it can't
        find enough features to track (caller should treat False as no-lock)."""
        try:
            self._dt = DroTrack(self._gray(frame), [float(v) for v in box])
            return True
        except Exception:
            self._dt = None
            return False

    def update(self, frame):
        if self._dt is None:
            return False, (0.0, 0.0, 0.0, 0.0)
        try:
            # FCM segmentation can hit benign divide-by-zero on flat regions
            with np.errstate(divide="ignore", invalid="ignore"):
                bbox, _center, _ext = self._dt.track(self._gray(frame))
            x, y, w, h = (float(v) for v in bbox)
            if w <= 0 or h <= 0 or not np.isfinite([x, y, w, h]).all():
                return False, (0.0, 0.0, 0.0, 0.0)
            return True, (x, y, w, h)
        except Exception:
            return False, (0.0, 0.0, 0.0, 0.0)
