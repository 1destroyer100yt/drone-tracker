"""Tests for occlusion coasting + re-identification (appearance.py + the
ObjectFollower state machine).

  python3 test_reid.py
"""

import cv2
import numpy as np

import appearance
import tracker as T
from detector import _DetectorBase, Detection

RED = (0, 0, 200)
BLUE = (200, 0, 0)


def scene(cx, cy=240, color=RED, size=50, bg=60):
    """A solid coloured square (with a white centre for CSRT texture) on a
    flat background. Returns (frame, box)."""
    f = np.full((480, 640, 3), bg, np.uint8)
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    cv2.rectangle(f, (x0, y0), (x0 + size, y0 + size), color, -1)
    cv2.circle(f, (int(cx), int(cy)), 7, (255, 255, 255), -1)
    return f, (x0, y0, size, size)


class StubDet(_DetectorBase):
    """Detector whose returns we set frame-by-frame; inherits match_in/pick_at
    /best_match from the real base so the follower behaves normally."""

    def __init__(self):
        self.conf, self.iou = 0.25, 0.45
        self.in_w = self.in_h = 640
        self.names = {3: "car", 4: "van"}
        self._dets = []

    def set(self, dets):
        self._dets = dets

    def detect(self, frame, classes=None):
        ds = self._dets
        if classes is not None:
            classes = set(classes)
            ds = [d for d in ds if d.cls in classes]
        return list(ds)


def car(box, cls=3):
    return Detection(box[0], box[1], box[2], box[3], 0.9, cls, "car")


def test_appearance():
    r, _ = scene(320, color=RED)
    b, _ = scene(320, color=BLUE)
    box = (295, 215, 50, 50)
    hr, hr2 = appearance.box_histogram(r, box), appearance.box_histogram(r, box)
    hb = appearance.box_histogram(b, box)
    same, diff = appearance.similarity(hr, hr2), appearance.similarity(hr, hb)
    assert same > 0.9 and diff < 0.5, (same, diff)
    print(f"appearance: OK  (same={same:.2f}, red-vs-blue={diff:.2f})")


def test_reid_through_occlusion():
    det = StubDet()
    fol = T.ObjectFollower(detector=det, target_classes={3}, detect_interval=1,
                           max_detect_misses=2, coast_seconds=15.0,
                           reid_min_sim=0.35)
    # lock a red car and let it drive right while visible
    f, box = scene(200)
    det.set([car(box)])
    assert fol.start(f, 200, 240) and fol.active
    t = 0.0
    for i in range(1, 8):
        t += 0.1
        f, box = scene(200 + i * 8)
        det.set([car(box)])
        fol.update(f, t)
    assert fol.active, fol.state
    lastx = 200 + 7 * 8

    # OCCLUSION: object hidden, detector sees nothing -> should start coasting
    for _ in range(5):
        t += 0.1
        det.set([])
        fol.update(np.full((480, 640, 3), 60, np.uint8), t)
    assert fol.coasting, fol.state

    # a BLUE car appears where the red one is predicted -> must NOT re-id it
    t += 0.1
    f, box = scene(lastx + 50, color=BLUE)
    det.set([car(box)])
    fol.update(f, t)
    assert fol.coasting, f"grabbed the wrong-colour car ({fol.state})"

    # the RED car reappears near the prediction -> re-identify and resume
    t += 0.1
    f, box = scene(lastx + 50, color=RED)
    det.set([car(box)])
    fol.update(f, t)
    assert fol.active, f"failed to re-identify ({fol.state})"
    print("re-id through occlusion: OK  (coasted, rejected blue, re-locked red)")


def test_coast_timeout():
    """After coast_seconds with no re-id, the target is truly LOST."""
    det = StubDet()
    fol = T.ObjectFollower(detector=det, target_classes={3}, detect_interval=1,
                           max_detect_misses=2, coast_seconds=1.0)
    f, box = scene(200)
    det.set([car(box)])
    fol.start(f, 200, 240)
    t = 0.0
    det.set([])
    blank = np.full((480, 640, 3), 60, np.uint8)
    for _ in range(40):                 # 4 s of nothing at 0.1 s steps
        t += 0.1
        fol.update(blank, t)
    assert not fol.alive, fol.state
    print("coast timeout: OK  (declared LOST after the window)")


def main():
    test_appearance()
    test_reid_through_occlusion()
    test_coast_timeout()
    print("\nall re-id tests passed.")


if __name__ == "__main__":
    main()
