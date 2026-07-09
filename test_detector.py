"""Tests for the ONNX detector and detection-assisted following.

  python3 test_detector.py

Covers:
  - IoU / geometry helpers
  - the real VisDrone ONNX model loads, has 10 classes, and runs a frame
    without error (skipped if the model file isn't present)
  - ObjectFollower snaps a click onto the detected box (via a stub detector)
  - confidence-gated loss: when the detector stops finding the object, the
    follower declares it LOST even while the CSRT tracker still "sees" something
    (the silent-drift fix)
"""

import math
import os

import cv2
import numpy as np

import tracker as T
from detector import Detection, iou_xywh, box_center

MODEL = os.path.join(os.path.dirname(__file__), "models", "visdrone_n.onnx")


def render(cx, cy, s=54):
    """A textured light square on a dark field, so CSRT has features to lock."""
    f = np.full((480, 640, 3), 30, np.uint8)
    x0, y0 = int(cx - s / 2), int(cy - s / 2)
    cv2.rectangle(f, (x0, y0), (x0 + s, y0 + s), (170, 170, 175), -1)
    for dx in range(-s // 2 + 6, s // 2 - 5, 9):
        for dy in range(-s // 2 + 6, s // 2 - 5, 9):
            cv2.circle(f, (int(cx + dx), int(cy + dy)), 2, (70, 70, 70), -1)
    return f, (x0, y0, s, s)


class StubDetector:
    """Stands in for YoloOnnxDetector: reports the object at a given box until
    `alive` is turned off, after which it finds nothing."""

    def __init__(self, box, cls=3):
        self.box = box
        self.cls = cls
        self.alive = True
        self.names = {cls: "car"}

    def _det(self):
        x, y, w, h = self.box
        return Detection(x, y, w, h, 0.9, self.cls, "car")

    def pick_at(self, frame, cx, cy, classes=None):
        return self._det() if self.alive else None

    def best_match(self, frame, ref_box, classes=None, min_iou=0.2):
        return self._det() if self.alive else None


def approx(a, b, tol=2.0):
    return abs(a - b) <= tol


def test_geometry():
    assert approx(iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)), 1.0, 1e-6)
    assert approx(iou_xywh((0, 0, 10, 10), (20, 20, 10, 10)), 0.0, 1e-6)
    # half-overlap in x, full in y -> IoU = 50/150
    assert approx(iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3.0, 1e-3)
    assert box_center((0, 0, 10, 20)) == (5.0, 10.0)
    print("geometry helpers: OK")


def test_snap_on_start():
    box = (300, 210, 60, 60)
    frame, _ = render(330, 240)
    det = StubDetector(box, cls=3)
    fol = T.ObjectFollower(detector=det, detect_interval=1)
    # click a bit off-centre; it should snap to the detected box
    assert fol.start(frame, 345, 250) is True
    assert fol.cls == 3
    fx, fy, fw, fh = fol.box
    assert approx(fx, box[0]) and approx(fy, box[1]), (fol.box, box)
    assert approx(fw, box[2]) and approx(fh, box[3]), (fol.box, box)
    print("snap-on-start: OK  (locked to detected box, class=car)")


def test_confidence_gated_loss():
    det = StubDetector((300, 210, 54, 54), cls=3)
    fol = T.ObjectFollower(detector=det, detect_interval=1, max_detect_misses=3)
    frame, _ = render(327, 237)
    assert fol.start(frame, 327, 237) is True

    # object present: follower stays locked across moving frames
    for i in range(5):
        f, box = render(327 + i * 3, 237)
        det.box = box
        c = fol.update(f)
        assert fol.active, f"dropped early at frame {i}"
    assert c is not None
    print("locked while detected: OK")

    # object gone from the detector; CSRT may still 'see' the square, but the
    # follower must declare loss within max_detect_misses detect cycles
    det.alive = False
    lost_at = None
    for i in range(10):
        f, _ = render(342 + i * 3, 237)  # square still on screen for CSRT
        c = fol.update(f)
        if not fol.active:
            lost_at = i
            break
    assert lost_at is not None, "never reported loss (silent drift!)"
    assert lost_at <= 3, f"took too long to report loss: {lost_at}"
    assert c is None
    print(f"confidence-gated loss: OK  (declared LOST after {lost_at + 1} "
          f"detect misses, despite CSRT still tracking)")


def test_real_model():
    if not os.path.exists(MODEL):
        print(f"real model: SKIP ({MODEL} not present)")
        return
    from detector import YoloOnnxDetector
    d = YoloOnnxDetector(MODEL)
    assert len(d.names) == 10, d.names
    assert d.class_id("car") == 3, d.names
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    dets = d.detect(frame)                 # noise -> probably empty, must not crash
    assert isinstance(dets, list)
    for det in dets:
        assert 0 <= det.cls < 10 and det.conf >= d.conf
    print(f"real model: OK  (10 classes, ran a frame, {len(dets)} spurious dets)")


def test_nms_format_output():
    """The end-to-end (nms=True) export gives [1, N, 6] = x1,y1,x2,y2,conf,cls.
    Feed a crafted one through the real detector's session and check it decodes
    and un-letterboxes correctly (this is the yolov8m export format)."""
    if not os.path.exists(MODEL):
        print("nms-format output: SKIP (model not present)")
        return
    from detector import YoloOnnxDetector
    d = YoloOnnxDetector(MODEL)

    class FakeSess:
        def __init__(self, out):
            self.out = out
        def run(self, *a, **k):
            return [self.out]

    # 640x480 frame -> letterbox r=1.0, px=0, py=80. A car box at 640-space
    # (50,60)-(150,160), plus a zero-padded row that must be dropped.
    out = np.array([[[50, 60, 150, 160, 0.9, 3],
                     [0, 0, 0, 0, 0, 0]]], np.float32)
    d.session = FakeSess(out)
    dets = d.detect(np.zeros((480, 640, 3), np.uint8))
    assert len(dets) == 1, dets
    det = dets[0]
    assert det.name == "car" and approx(det.conf, 0.9, 1e-4)
    assert approx(det.x, 50) and approx(det.y, -20)      # (60 - py=80)/r
    assert approx(det.w, 100) and approx(det.h, 100)
    # class filter still works on this format
    assert d.detect(np.zeros((480, 640, 3), np.uint8), classes={0}) == []
    print("nms-format output: OK  (end2end [N,6] decoded + unletterboxed)")


def test_default_model_path():
    """--detector with no path prefers yolov8m, falls back to the bundled nano."""
    import tempfile
    from detector import default_model_path
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "visdrone_n.onnx"), "w").close()
        assert default_model_path(td).endswith("visdrone_n.onnx")
        open(os.path.join(td, "visdrone_m.onnx"), "w").close()
        assert default_model_path(td).endswith("visdrone_m.onnx")
    print("default_model_path: OK  (prefers yolov8m, falls back to nano)")


def test_best_match_distance_gate():
    """A far-away same-class detection must NOT be re-locked onto (returns None),
    so a vanished object triggers loss instead of an identity switch."""
    if not os.path.exists(MODEL):
        print("best_match distance gate: SKIP (model not present)")
        return
    from detector import YoloOnnxDetector

    class FakeSess:
        def __init__(self, out):
            self.out = out
        def run(self, *a, **k):
            return [self.out]

    d = YoloOnnxDetector(MODEL)
    # one car far in the corner (640-space), no letterbox pad for 640x640
    d.session = FakeSess(np.array([[[600, 600, 630, 630, 0.9, 3]]], np.float32))
    frame = np.zeros((640, 640, 3), np.uint8)
    ref_near = (600, 600, 30, 30)        # overlaps the detection -> match
    ref_far = (10, 10, 30, 30)           # far away -> should be rejected
    assert d.best_match(frame, ref_near) is not None
    assert d.best_match(frame, ref_far) is None
    print("best_match distance gate: OK  (rejects distant distractor)")


COREML = os.path.join(os.path.dirname(__file__), "models", "visdrone_m.mlpackage")


def test_build_detector_dispatch():
    """build_detector picks the backend by extension."""
    from detector import build_detector, YoloOnnxDetector
    if os.path.exists(MODEL):
        assert isinstance(build_detector(MODEL), YoloOnnxDetector)
    print("build_detector dispatch: OK")


def test_coreml_backend():
    """CoreML detector loads, shares the base interface, and runs (macOS +
    coremltools + the .mlpackage required; skipped otherwise)."""
    try:
        import coremltools  # noqa: F401
    except Exception:
        print("coreml backend: SKIP (coremltools not installed)")
        return
    if not os.path.exists(COREML):
        print("coreml backend: SKIP (models/visdrone_m.mlpackage not present)")
        return
    from detector import build_detector
    from coreml_detector import CoreMLDetector
    d = build_detector(COREML)
    assert isinstance(d, CoreMLDetector)
    assert len(d.names) == 10 and d.class_id("car") == 3
    # runs without crashing; black frame -> a list (usually empty)
    out = d.detect(np.zeros((480, 640, 3), np.uint8))
    assert isinstance(out, list)
    # shares the selection logic from the base
    assert hasattr(d, "best_match") and hasattr(d, "pick_at")
    print(f"coreml backend: OK  (ANE, {len(d.names)} classes)")


def test_parse_tiles():
    from detector import parse_tiles
    assert parse_tiles(None) is None and parse_tiles("") is None
    assert parse_tiles("2x3") == (2, 3)
    assert parse_tiles("4") == (4, 4)
    print("parse_tiles: OK")


def test_tiled_detector():
    """TiledDetector wraps a base detector, keeps the shared interface, and
    runs (uses the bundled nano ONNX; skipped if absent)."""
    if not os.path.exists(MODEL):
        print("tiled detector: SKIP (model not present)")
        return
    from detector import build_detector, TiledDetector, parse_tiles
    d = build_detector(MODEL, conf=0.25, tiles=parse_tiles("2x2"))
    assert isinstance(d, TiledDetector)
    assert d.class_id("car") == 3 and len(d.names) == 10
    out = d.detect(np.zeros((480, 640, 3), np.uint8))
    assert isinstance(out, list)
    assert hasattr(d, "best_match") and hasattr(d, "pick_at")
    print("tiled detector: OK  (2x2 grid, shares base interface)")


def main():
    test_geometry()
    test_snap_on_start()
    test_confidence_gated_loss()
    test_real_model()
    test_nms_format_output()
    test_default_model_path()
    test_best_match_distance_gate()
    test_build_detector_dispatch()
    test_coreml_backend()
    test_parse_tiles()
    test_tiled_detector()
    print("\nall detector tests passed.")


if __name__ == "__main__":
    main()
