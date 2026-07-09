"""YOLOv8 detector on the Apple Neural Engine via CoreML (macOS only).

Runs a `.mlpackage` exported by Ultralytics (e.g. `models/visdrone_m.mlpackage`)
through CoreML, which schedules it on the Neural Engine -- far faster than the
ONNX-on-CPU path for the bigger yolov8m, enough for real-time on an M-series Mac.

The Ultralytics CoreML export bakes in NMS: the model takes an RGB image plus
`iouThreshold`/`confidenceThreshold`, and returns `confidence` (N x num_classes)
and `coordinates` (N x 4, normalized cx,cy,w,h relative to the letterboxed input).
We letterbox the frame, predict, then un-letterbox the boxes back to pixels.

Presents the same interface as YoloOnnxDetector (detect / best_match / pick_at /
class_id / names) via the shared _DetectorBase, so it drops into ObjectFollower
unchanged.  Requires: `pip install coremltools pillow`.
"""

import cv2
import numpy as np

from detector import _DetectorBase


class CoreMLDetector(_DetectorBase):
    def __init__(self, model_path, names_path=None, conf=0.25, iou=0.45,
                 compute_units=None):
        try:
            import coremltools as ct
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "coremltools is required for the CoreML detector "
                "(`pip install coremltools pillow`)") from e

        self.conf = conf
        self.iou = iou
        cu = compute_units or ct.ComputeUnit.ALL       # lets CoreML use the ANE
        self.model = ct.models.MLModel(model_path, compute_units=cu)

        # discover the image input name + size and the threshold input names
        self.image_input = "image"
        self.in_w = self.in_h = 640
        self._iou_key = self._conf_key = None
        for i in self.model.get_spec().description.input:
            kind = i.type.WhichOneof("Type")
            if kind == "imageType":
                self.image_input = i.name
                self.in_w = i.type.imageType.width or 640
                self.in_h = i.type.imageType.height or 640
            elif kind == "doubleType":
                low = i.name.lower()
                if "iou" in low:
                    self._iou_key = i.name
                elif "conf" in low:
                    self._conf_key = i.name
        self.names = self._load_names(names_path, model_path)

    def detect(self, frame, classes=None):
        """Detect objects in a BGR frame -> list[Detection], highest conf first."""
        from PIL import Image
        canvas, r, px, py = self._letterbox(frame)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        feed = {self.image_input: pil}
        if self._iou_key:
            feed[self._iou_key] = float(self.iou)
        if self._conf_key:
            feed[self._conf_key] = float(self.conf)
        out = self.model.predict(feed)

        conf = np.asarray(out["confidence"], dtype=np.float32)   # [N, nc]
        coord = np.asarray(out["coordinates"], dtype=np.float32)  # [N, 4] norm
        if conf.size == 0 or coord.size == 0:
            return []

        cls_ids = np.argmax(conf, axis=1)
        confs = conf[np.arange(conf.shape[0]), cls_ids]
        classes = set(int(c) for c in classes) if classes is not None else None

        # normalized (0..1) cx,cy,w,h in the letterboxed input -> original px
        cx = coord[:, 0] * self.in_w
        cy = coord[:, 1] * self.in_h
        bw = coord[:, 2] * self.in_w
        bh = coord[:, 3] * self.in_h
        x = (cx - bw / 2.0 - px) / r
        y = (cy - bh / 2.0 - py) / r
        w = bw / r
        h = bh / r

        order = np.argsort(-confs)      # CoreML already NMS'd; just sort
        dets = []
        for i in order:
            c = float(confs[i])
            if c < self.conf:
                continue
            cid = int(cls_ids[i])
            if classes is not None and cid not in classes:
                continue
            dets.append(self._mk(x[i], y[i], w[i], h[i], c, cid))
        return dets
