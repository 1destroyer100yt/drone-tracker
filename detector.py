"""Lightweight YOLOv8 object detector via ONNX Runtime (no PyTorch).

Runs a YOLOv8 detect model exported to ONNX (e.g. our VisDrone-trained
`models/visdrone_n.onnx`) with just onnxruntime + numpy + OpenCV. This keeps the
follower deployable on a laptop or a Raspberry Pi without dragging in torch.

The follower uses it two ways (see ObjectFollower in tracker.py):
  - seed a click/box onto the *actual* detected object box (tighter than a
    fixed square), and
  - periodically re-lock the CSRT tracker to a fresh detection so it can't
    silently drift onto the background, and report a genuine loss when the
    object truly isn't detected any more (confidence-gated loss reporting).

onnxruntime is an optional dependency; importing this module without it raises a
clear error only when you actually try to build a detector.
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


def default_model_path(models_dir):
    """Best bundled detector available: prefer the bigger, more accurate
    yolov8m, else the yolov8n that ships in the repo. Used when --detector is
    given with no explicit path."""
    for name in ("visdrone_m.onnx", "visdrone_n.onnx"):
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            return p
    return os.path.join(models_dir, "visdrone_n.onnx")


class YoloOnnxDetector:
    """A YOLOv8 detect model running on ONNX Runtime.

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

    @staticmethod
    def _load_names(names_path, onnx_path):
        if names_path is None:
            guess = os.path.join(os.path.dirname(onnx_path),
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
        """Resize keeping aspect ratio, pad to the network size. Returns the
        tensor plus (scale, pad_x, pad_y) to map boxes back to the original."""
        h0, w0 = frame.shape[:2]
        r = min(self.in_w / w0, self.in_h / h0)
        nw, nh = round(w0 * r), round(h0 * r)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        px, py = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None]        # 1,3,H,W
        return np.ascontiguousarray(tensor), r, px, py

    def _mk(self, x, y, w, h, conf, cls):
        cls = int(cls)
        return Detection(float(x), float(y), float(w), float(h), float(conf),
                         cls, self.names.get(cls, str(cls)))

    def detect(self, frame, classes=None):
        """Detect objects in a BGR frame.

        Handles both YOLOv8 ONNX output layouts:
          - raw head (exported nms=False): [1, 4+nc, 8400] -> we decode + NMS
          - end-to-end (exported nms=True): [1, N, 6] = x1,y1,x2,y2,conf,cls,
            already NMS'd, so we just filter and rescale.

        classes: optional iterable of class ids to keep (others dropped).
        Returns a list of Detection, highest confidence first.
        """
        tensor, r, px, py = self._letterbox(frame)
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
        # nothing overlaps: the nearest-center detection can recover a
        # fast-moving object, but only if it's plausibly close.
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
