"""Lightweight YOLOv8 object detectors for the follower.

Two backends with an identical interface (see ObjectFollower in tracker.py):
  - `YoloOnnxDetector` -- ONNX Runtime, no PyTorch: runs anywhere (laptop, Pi).
  - `CoreMLDetector`   -- Apple Neural Engine via CoreML (macOS only, much
    faster for the bigger yolov8m); lives in `coreml_detector.py`.

Use `build_detector(path)` to get the right one by file extension, or
`default_model_path(models_dir)` + `build_detector` for the auto default.

The follower uses a detector two ways: seed a click/box onto the *actual*
detected object box, and periodically re-lock the tracker to a fresh detection
so it can't silently drift -- reporting a genuine loss when the object is gone.

onnxruntime / coremltools are optional deps; the error is raised only when you
actually build that backend.
"""

import json
import math
import os
from collections import namedtuple

import cv2
import numpy as np

# (x, y, w, h) in original-image pixels, plus confidence, class id and name.
Detection = namedtuple("Detection", "x y w h conf cls name")


def iou_xywh(a, b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    x, y, w, h = box[:4]
    return (x + w / 2.0, y + h / 2.0)


def _coreml_available():
    try:
        import coremltools  # noqa: F401
        return True
    except Exception:
        return False


def default_model_path(models_dir):
    """Best bundled detector available, in preference order. On a Mac with
    coremltools we prefer the yolov8m CoreML package (Neural Engine, real-time);
    otherwise the yolov8m ONNX if present, else the in-repo yolov8n. Used when
    --detector is given with no explicit path."""
    import platform
    names = []
    if platform.system() == "Darwin" and _coreml_available():
        names.append("visdrone_m.mlpackage")
    names += ["visdrone_m.onnx", "visdrone_n.onnx"]
    for name in names:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            return p
    return os.path.join(models_dir, "visdrone_n.onnx")


def build_detector(path, conf=0.25, iou=0.45, tiles=None):
    """Construct the right detector backend for a model path by extension:
    .mlpackage/.mlmodel -> CoreML (ANE), anything else -> ONNX Runtime. If
    `tiles` (cols, rows) is given, wrap it in a TiledDetector for high-res
    frames."""
    if path.endswith((".mlpackage", ".mlmodel")):
        from coreml_detector import CoreMLDetector
        base = CoreMLDetector(path, conf=conf, iou=iou)
    else:
        base = YoloOnnxDetector(path, conf=conf, iou=iou)
    return TiledDetector(base, tiles=tiles) if tiles else base


def parse_tiles(spec):
    """'2x3' -> (2, 3) cols x rows; 'N' -> (N, N); None/'' -> None."""
    if not spec:
        return None
    spec = str(spec).lower().replace(" ", "")
    if "x" in spec:
        c, r = spec.split("x")
        return (int(c), int(r))
    n = int(spec)
    return (n, n)


class _DetectorBase:
    """Shared geometry + selection logic on top of a subclass `detect()`.

    Subclasses set: self.conf, self.iou, self.in_w, self.in_h, self.names, and
    implement detect(frame, classes) -> list[Detection]."""

    @staticmethod
    def _load_names(names_path, model_path):
        """Class-id -> name map, from an explicit json or a sibling
        visdrone_classes.json next to the model."""
        if names_path is None:
            guess = os.path.join(os.path.dirname(model_path),
                                 "visdrone_classes.json")
            names_path = guess if os.path.exists(guess) else None
        if names_path and os.path.exists(names_path):
            raw = json.load(open(names_path))
            return {int(k): v for k, v in raw.items()}
        return {}

    def class_id(self, name):
        """Look up a class id by (case-insensitive) name, or None."""
        for cid, cname in self.names.items():
            if cname.lower() == str(name).lower():
                return cid
        return None

    def _letterbox(self, frame):
        """Resize keeping aspect ratio and pad to the network size. Returns the
        padded BGR canvas plus (scale, pad_x, pad_y) to map boxes back."""
        h0, w0 = frame.shape[:2]
        r = min(self.in_w / w0, self.in_h / h0)
        nw, nh = round(w0 * r), round(h0 * r)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        px, py = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, r, px, py

    def _mk(self, x, y, w, h, conf, cls):
        cls = int(cls)
        return Detection(float(x), float(y), float(w), float(h), float(conf),
                         cls, self.names.get(cls, str(cls)))

    def best_match(self, frame, ref_box, classes=None, min_iou=0.2,
                   max_center_dist=None):
        """Return the detection that best matches ref_box: the highest-IoU one
        above min_iou, or -- if none overlap -- the nearest-center detection,
        but only if it's within max_center_dist (default 2.5x the ref-box
        diagonal). Returns None otherwise, so a vanished object is NOT re-locked
        onto a distant distractor of the same class (which would defeat loss
        reporting and cause identity switches on crowded scenes)."""
        dets = self.detect(frame, classes=classes)
        if not dets:
            return None
        scored = [(iou_xywh(d, ref_box), d) for d in dets]
        best_iou, best = max(scored, key=lambda s: s[0])
        if best_iou >= min_iou:
            return best
        rcx, rcy = box_center(ref_box)
        nearest = min(dets, key=lambda d: (box_center(d)[0] - rcx) ** 2
                      + (box_center(d)[1] - rcy) ** 2)
        if max_center_dist is None:
            diag = math.hypot(ref_box[2], ref_box[3])
            max_center_dist = 2.5 * max(diag, 1.0)
        ncx, ncy = box_center(nearest)
        if math.hypot(ncx - rcx, ncy - rcy) <= max_center_dist:
            return nearest
        return None

    def pick_at(self, frame, cx, cy, classes=None):
        """Pick the detection to lock onto for a click at (cx, cy): a box that
        contains the point (smallest such box), else the nearest-center one."""
        dets = self.detect(frame, classes=classes)
        if not dets:
            return None
        inside = [d for d in dets
                  if d.x <= cx <= d.x + d.w and d.y <= cy <= d.y + d.h]
        if inside:
            return min(inside, key=lambda d: d.w * d.h)
        return min(dets, key=lambda d: (box_center(d)[0] - cx) ** 2
                   + (box_center(d)[1] - cy) ** 2)


class YoloOnnxDetector(_DetectorBase):
    """A YOLOv8 detect model running on ONNX Runtime (CPU by default).

    detect(frame) -> list[Detection] in the frame's own pixel coordinates.
    """

    def __init__(self, onnx_path, names_path=None, conf=0.25, iou=0.45,
                 providers=None):
        try:
            import onnxruntime as ort
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "onnxruntime is required for the ONNX detector "
                "(`pip install onnxruntime`)") from e
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"detector model not found: {onnx_path}")

        self.conf = conf
        self.iou = iou
        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # static [1,3,H,W]; fall back to 640 if a dim is dynamic
        _, _, h, w = inp.shape
        self.in_h = int(h) if isinstance(h, int) else 640
        self.in_w = int(w) if isinstance(w, int) else 640
        self.names = self._load_names(names_path, onnx_path)

    def detect(self, frame, classes=None):
        """Detect objects in a BGR frame. Handles both ONNX output layouts:
          - raw head (nms=False): [1, 4+nc, 8400] -> we decode + NMS
          - end-to-end (nms=True): [1, N, 6] = x1,y1,x2,y2,conf,cls (pre-NMS'd).
        classes: optional iterable of class ids to keep. Highest conf first."""
        canvas, r, px, py = self._letterbox(frame)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        out = self.session.run(None, {self.input_name: tensor})[0]
        arr = np.squeeze(out, 0)
        classes = set(int(c) for c in classes) if classes is not None else None

        # --- end-to-end (NMS baked in): [N, 6] = x1,y1,x2,y2,conf,cls ---
        if arr.ndim == 2 and arr.shape[1] == 6 and arr.shape[0] != 6:
            confs = arr[:, 4].astype(np.float32)
            cls_ids = arr[:, 5].astype(np.int32)
            keep = confs >= self.conf
            if classes is not None:
                keep &= np.array([int(c) in classes for c in cls_ids])
            if not np.any(keep):
                return []
            b = arr[keep]
            confs = confs[keep]
            cls_ids = cls_ids[keep]
            x = (b[:, 0] - px) / r
            y = (b[:, 1] - py) / r
            w = (b[:, 2] - b[:, 0]) / r
            h = (b[:, 3] - b[:, 1]) / r
            order = np.argsort(-confs)          # already NMS'd; sort by conf
            return [self._mk(x[i], y[i], w[i], h[i], confs[i], cls_ids[i])
                    for i in order]

        # --- raw head: [4+nc, 8400] (or transposed); decode + NMS ourselves ---
        preds = arr.T if arr.shape[0] < arr.shape[1] else arr
        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]
        cls_ids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(scores_all.shape[0]), cls_ids]

        keep = confs >= self.conf
        if classes is not None:
            keep &= np.array([int(c) in classes for c in cls_ids])
        if not np.any(keep):
            return []
        boxes_xywh = boxes_xywh[keep]
        confs = confs[keep]
        cls_ids = cls_ids[keep]

        # center-xywh (letterboxed px) -> top-left xywh in original image px
        cx, cy, bw, bh = boxes_xywh.T
        x = (cx - bw / 2.0 - px) / r
        y = (cy - bh / 2.0 - py) / r
        w = bw / r
        h = bh / r
        rects = np.stack([x, y, w, h], axis=1)

        idx = cv2.dnn.NMSBoxes(rects.tolist(), confs.tolist(),
                               self.conf, self.iou)
        if len(idx) == 0:
            return []
        idx = np.array(idx).flatten()
        dets = [self._mk(x[i], y[i], w[i], h[i], confs[i], cls_ids[i])
                for i in idx]
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets


class TiledDetector(_DetectorBase):
    """Wraps a base detector and runs it on an overlapping grid of tiles plus a
    full-frame pass, merging results with per-class NMS. Recovers small objects
    in high-resolution frames that get lost when the whole frame is squished to
    the network's 640x640 input. Cost scales with the tile count (tiles + 1
    detector calls per frame), so it's a quality/speed trade for high-altitude
    or 4K footage rather than a real-time default."""

    def __init__(self, base, tiles=(2, 2), overlap=0.2):
        self.base = base
        self.names = base.names
        self.conf = base.conf
        self.iou = base.iou
        self.in_w = base.in_w
        self.in_h = base.in_h
        self.cols, self.rows = tiles
        self.overlap = overlap

    def class_id(self, name):
        return self.base.class_id(name)

    def detect(self, frame, classes=None):
        h, w = frame.shape[:2]
        dets = list(self.base.detect(frame, classes=classes))   # full frame
        tw, th = w / self.cols, h / self.rows
        ox, oy = tw * self.overlap, th * self.overlap
        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0 = int(max(0, c * tw - ox)), int(max(0, r * th - oy))
                x1 = int(min(w, (c + 1) * tw + ox))
                y1 = int(min(h, (r + 1) * th + oy))
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                for d in self.base.detect(crop, classes=classes):
                    dets.append(d._replace(x=d.x + x0, y=d.y + y0))
        if not dets:
            return []
        # per-class NMS to merge duplicates from the overlapping tiles
        final = []
        for cls in set(d.cls for d in dets):
            g = [d for d in dets if d.cls == cls]
            boxes = [[d.x, d.y, d.w, d.h] for d in g]
            scores = [d.conf for d in g]
            idx = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
            for i in (np.array(idx).flatten() if len(idx) else []):
                final.append(g[i])
        final.sort(key=lambda d: d.conf, reverse=True)
        return final
