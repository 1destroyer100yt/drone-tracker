"""Lightweight appearance signature for re-identifying a tracked object after
an occlusion -- an HSV colour histogram, no deep-learning embedding, so it stays
Raspberry-Pi friendly.

Hue+Saturation histograms are robust to brightness changes (a car keeps its
colour through shadow), which is what we need to decide "is this the same vehicle
we lost a few seconds ago?".
"""

import cv2
import numpy as np

H_BINS, S_BINS = 30, 32


def crop(frame, box, pad=0.0):
    """Return the image patch under an (x, y, w, h) box (optionally padded),
    or None if it falls outside the frame."""
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x0 = int(max(0, x - bw * pad))
    y0 = int(max(0, y - bh * pad))
    x1 = int(min(w, x + bw * (1 + pad)))
    y1 = int(min(h, y + bh * (1 + pad)))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def histogram(patch):
    """Normalised H-S colour histogram for an image patch, or None."""
    if patch is None or patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS],
                        [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def box_histogram(frame, box):
    """Convenience: histogram of the patch under a box."""
    return histogram(crop(frame, box))


def similarity(h1, h2):
    """Colour-histogram similarity in [0, 1] (correlation; higher = more alike)."""
    if h1 is None or h2 is None:
        return 0.0
    return float(max(0.0, cv2.compareHist(np.asarray(h1, np.float32),
                                          np.asarray(h2, np.float32),
                                          cv2.HISTCMP_CORREL)))
