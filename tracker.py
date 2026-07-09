"""Body and face tracker (MediaPipe, Raspberry Pi friendly).

Opens the webcam and tracks people using a single MediaPipe pose network:

  - RED cross on each detected face (from the pose face landmarks) and on the
    body (torso center)
  - GREEN cross on the person closest to the center of the screen
  - BLUE cross fixed in the middle of the screen
  - a line from the blue cross to every red cross, labeled with the
    Euclidean distance in pixels:  d = sqrt(dx^2 + dy^2)

Accuracy features:
  - One-Euro filter on every cross: kills jitter when you hold still,
    adds almost no lag when you move fast (better than a plain average)
  - face and torso centers weighted by per-landmark visibility
  - video-mode tracking between frames instead of re-detecting from scratch

Performance features (for Raspberry Pi / similar):
  - ONE neural net: face point comes from the pose landmarks, so there is no
    separate face-detector inference (roughly halves the per-frame model work)
  - threaded capture that always hands over the newest frame and drops stale
    ones, so the gimbal aims at where the person is now, not where they were
  - MJPG capture + 1-frame driver buffer for higher FPS and lower latency
  - --model lite|full, auto-picks lite on ARM boards
  - --headless to run with no display (onboard the aircraft, over SSH, etc.)
  - FPS counter on screen (and printed periodically in --headless)

UAV (ArduPilot):
  With --mavlink the aircraft's CAMERA GIMBAL is aimed at the tracked GREEN
  target via MOUNT_CONTROL. It moves the camera only -- it never steers the
  plane, arms, changes mode, or overrides RC. See uav.py.

Usage:
  python3 tracker.py                 # defaults (full model on desktop)
  python3 tracker.py --model lite    # force the lite pose model
  python3 tracker.py --width 320 --height 240   # extra headroom on a Pi
  python3 tracker.py --headless --mavlink /dev/ttyAMA0   # onboard, no screen
  python3 tracker.py --num-poses 3   # track up to 3 people

Press 'q' or Esc to quit (Ctrl+C in --headless).
"""

import argparse
import math
import os
import platform
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import distance
import size
from filters import OneEuroFilter, PointSmoother
from motion import VelocityTracker, real_speed

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
POSE_MODELS = {
    "lite": os.path.join(MODEL_DIR, "pose_landmarker_lite.task"),
    "full": os.path.join(MODEL_DIR, "pose_landmarker_full.task"),
}

CROSS_SIZE = 20
RED = (0, 0, 255)     # BGR
BLUE = (255, 0, 0)    # BGR
GREEN = (0, 255, 0)   # BGR
WHITE = (255, 255, 255)

# pose landmark indices
FACE_LANDMARKS = (0, 2, 5, 7, 8)          # nose, eyes, ears
TORSO_LANDMARKS = (11, 12, 23, 24)        # shoulders, hips


def on_arm_board():
    """True on Raspberry Pi and similar ARM Linux boards."""
    return platform.system() == "Linux" and platform.machine() in (
        "armv7l", "aarch64", "arm64")


def weighted_center(landmarks, indices, w, h, vis_thresh=0.5):
    """Visibility-weighted pixel center of the given pose landmarks, or None
    if none of them are confidently visible."""
    pts = [landmarks[j] for j in indices if landmarks[j].visibility > vis_thresh]
    if not pts:
        return None
    total = sum(p.visibility for p in pts)
    cx = sum(p.x * p.visibility for p in pts) / total * w
    cy = sum(p.y * p.visibility for p in pts) / total * h
    return cx, cy


def draw_cross(frame, x, y, color, size=CROSS_SIZE, thickness=2):
    x, y = int(round(x)), int(round(y))
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


def draw_distance(frame, center, target):
    """Line from the blue center cross to a red cross, labeled with the
    Euclidean distance in pixels."""
    cx, cy = center
    tx, ty = target
    dist = math.hypot(tx - cx, ty - cy)  # sqrt(dx^2 + dy^2)

    p1 = (int(round(cx)), int(round(cy)))
    p2 = (int(round(tx)), int(round(ty)))
    cv2.line(frame, p1, p2, WHITE, 1, cv2.LINE_AA)

    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    label = f"{dist:.0f}px"
    cv2.putText(frame, label, (mid[0] + 6, mid[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (mid[0] + 6, mid[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


class TrackAssigner:
    """Gives each detection a STABLE id across frames by matching it to the
    nearest detection of the same kind last frame. MediaPipe doesn't guarantee
    pose ordering, so without this the per-track smoothing filters would swap
    between people when more than one is in view."""

    def __init__(self, max_dist_frac=0.15):
        self.max_dist_frac = max_dist_frac
        self.prev = {}       # key -> (x, y)
        self.next_id = 0

    def assign(self, dets, w, h):
        """dets: list of (x, y, label, ...). Returns a list of keys aligned to
        dets, matching within the same label by nearest previous position."""
        maxd = self.max_dist_frac * math.hypot(w, h)
        maxd2 = maxd * maxd
        prev_items = list(self.prev.items())
        pairs = []
        for di, d in enumerate(dets):
            x, y, label = d[0], d[1], d[2]
            for k, (px, py) in prev_items:
                if not k.startswith(label):
                    continue
                dd = (x - px) ** 2 + (y - py) ** 2
                if dd <= maxd2:
                    pairs.append((dd, di, k))
        pairs.sort()
        taken_det, taken_key = {}, set()
        for dd, di, k in pairs:
            if di in taken_det or k in taken_key:
                continue
            taken_det[di] = k
            taken_key.add(k)
        keys, new_prev = [], {}
        for di, d in enumerate(dets):
            k = taken_det.get(di)
            if k is None:
                k = f"{d[2]}{self.next_id}"
                self.next_id += 1
            keys.append(k)
            new_prev[k] = (d[0], d[1])
        self.prev = new_prev
        return keys


class ObjectFollower:
    """Follows a single clicked object (a car, a bag, anything) with an OpenCV
    CSRT tracker. Seeded from a click (a default-size box) or a drawn box, it
    reports the box centre each frame and tolerates brief occlusions before
    giving up. CSRT is the accurate OpenCV tracker; use KCF on weak CPUs.

    Optionally takes a YoloOnnxDetector. With one, the follower gets far more
    reliable:
      - a click/box locks onto the *actual* detected object box, not a fixed
        square, and remembers that object's class;
      - every `detect_interval` frames (and whenever CSRT fails) it re-locks the
        tracker to a fresh detection of that class, so it can't slowly drift
        onto the background;
      - if the detector can't find the object for `max_detect_misses` detection
        cycles in a row, it declares the target LOST instead of confidently
        tracking nothing (confidence-gated loss reporting).
    Without a detector it behaves exactly as before (pure CSRT/KCF/DroTrack)."""

    def __init__(self, box_size=80, max_lost=20, algo="CSRT", detector=None,
                 detect_interval=15, match_iou=0.2, target_classes=None,
                 max_detect_misses=3):
        self.box_size = box_size
        self.max_lost = max_lost
        self.algo = algo
        self.detector = detector
        self.detect_interval = max(1, detect_interval)
        self.match_iou = match_iou
        self.target_classes = target_classes    # allowed class ids, or None
        self.max_detect_misses = max_detect_misses
        self.tracker = None
        self.box = None            # (x, y, w, h) ints
        self.lost = 0
        self.frame_i = 0
        self.det_misses = 0
        self.cls = None            # locked object's class id (from detector)
        self.score = None          # last detection confidence
        self.vel = None            # VelocityTracker (created on start)
        self.speed_px = 0.0        # smoothed target speed, pixels/second
        self.scene_scale = None    # metres-per-pixel, inferred from vehicles
        self.last_dets = []        # last full-scene detections (for size/re-id)

    @property
    def active(self):
        return self.tracker is not None

    def _new_tracker(self):
        if self.algo == "KCF" and hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if self.algo == "DROTRACK":
            from third_party.drotrack import DroTrackCV   # lazy: needs scipy
            return DroTrackCV()
        return cv2.TrackerCSRT_create()

    def _clip_box(self, box, w, h):
        x, y, bw, bh = box
        x = int(max(0, min(x, w - 2)))
        y = int(max(0, min(y, h - 2)))
        bw = int(max(8, min(bw, w - x)))
        bh = int(max(8, min(bh, h - y)))
        return x, y, bw, bh

    def _init_tracker(self, frame, box):
        """(Re)create the tracker locked to box. Returns True on success."""
        h, w = frame.shape[:2]
        x, y, bw, bh = self._clip_box(box, w, h)
        self.tracker = self._new_tracker()
        try:
            res = self.tracker.init(frame, (x, y, bw, bh))
        except Exception:
            res = False
        if res is False:        # DroTrack couldn't lock (cv2 returns None = ok)
            self.tracker = None
            self.box = None
            return False
        self.box = (x, y, bw, bh)
        return True

    def _cls_filter(self):
        if self.cls is not None:
            return {self.cls}
        return self.target_classes

    def start(self, frame, cx, cy, box=None):
        """Begin tracking at pixel (cx, cy), or an explicit (x,y,w,h) box.
        With a detector, snap onto the detected object under the click/box."""
        self.cls = None
        self.score = None
        if self.detector is not None:
            try:
                if box is not None:
                    det = self.detector.best_match(
                        frame, box, classes=self.target_classes, min_iou=0.1)
                else:
                    det = self.detector.pick_at(
                        frame, cx, cy, classes=self.target_classes)
            except Exception:
                det = None
            if det is not None:
                box = (det.x, det.y, det.w, det.h)
                self.cls = det.cls
                self.score = det.conf
        if box is None:
            s = self.box_size
            box = (cx - s / 2, cy - s / 2, s, s)
        if not self._init_tracker(frame, box):
            return False
        self.lost = 0
        self.frame_i = 0
        self.det_misses = 0
        self.vel = VelocityTracker()
        self.speed_px = 0.0
        self.scene_scale = None
        self.last_dets = []
        return True

    def _redetect(self, frame):
        """Try to re-lock onto a fresh detection of the tracked object.
        Returns the (cx, cy) centre on success, or None. On repeated failure it
        declares the target lost and clears. One full-scene detect here also
        feeds the scene scale (size) and the detection cache (re-id)."""
        try:
            dets = self.detector.detect(frame)
        except Exception:
            dets = []
        self.last_dets = dets
        sc = size.estimate_scene_scale(dets)
        if sc:
            self.scene_scale = sc
        cf = self._cls_filter()
        cand = dets if cf is None else [d for d in dets if d.cls in cf]
        match = self.detector.match_in(cand, self.box, min_iou=self.match_iou)
        if match is not None:
            if self._init_tracker(frame, (match.x, match.y, match.w, match.h)):
                self.lost = 0
                self.det_misses = 0
                self.score = match.conf
                x, y, bw, bh = self.box
                return (x + bw / 2.0, y + bh / 2.0)
        self.det_misses += 1
        if self.det_misses >= self.max_detect_misses:
            self.clear()
        return None

    def _track_velocity(self, center, t):
        if center is None or self.vel is None:
            return
        if t is None:
            t = time.monotonic()
        self.speed_px = self.vel.update(t, center[0], center[1])

    def update(self, frame, t=None):
        """Return the (cx, cy) centre of the tracked box, or None if lost.
        Pass `t` (seconds) for correct speed on recorded/variable-rate video;
        omit it and wall-clock time is used (correct for a live camera)."""
        if self.tracker is None:
            return None
        ok, box = self.tracker.update(frame)
        center = None
        if ok:
            self.box = tuple(int(v) for v in box)
            self.lost = 0
            x, y, bw, bh = self.box
            center = (x + bw / 2.0, y + bh / 2.0)
        else:
            self.lost += 1

        if self.detector is not None:
            self.frame_i += 1
            # re-lock on a schedule, or immediately if CSRT lost the object
            if not ok or self.frame_i % self.detect_interval == 0:
                relock = self._redetect(frame)
                if relock is not None:
                    center = relock
                elif self.tracker is None:    # _redetect declared it lost
                    self.speed_px = 0.0
                    return None
                # else: no match this cycle -> center stays (None between fixes)
        elif not ok and self.lost > self.max_lost:
            self.clear()                       # no detector: brief-occlusion only
            return None

        self._track_velocity(center, t)
        return center

    def clear(self):
        self.tracker = None
        self.box = None
        self.lost = 0
        self.frame_i = 0
        self.det_misses = 0
        self.cls = None
        self.score = None
        self.vel = None
        self.speed_px = 0.0
        self.scene_scale = None
        self.last_dets = []


class FrameGrabber:
    """Reads the camera in a background thread and always exposes the newest
    frame, dropping any the main loop was too busy to consume. Keeps end-to-end
    latency at one frame instead of letting the driver queue stale frames."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.stopped = True
                break
            with self.lock:
                self.frame = frame
                self.seq += 1

    def read(self):
        """Return (frame, seq). frame is None until the first one arrives.
        seq lets the caller skip re-processing a frame it already handled."""
        with self.lock:
            return self.frame, self.seq

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


def int_or_str(value):
    """--camera accepts an index (0, 1, ...) or an RTSP/HTTP URL / file path."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def open_camera(args):
    """Open a local camera by index, or a network stream / file by URL.
    Strings are treated as RTSP/HTTP/file sources via the FFMPEG backend."""
    src = args.camera
    is_index = isinstance(src, int)
    cap = cv2.VideoCapture(src) if is_index \
        else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera source: {src!r} "
                         "(check the index, URL, or camera permissions)")
    if is_index:
        # MJPG lifts the FPS cap many USB cams impose on raw YUYV; width/height
        # apply to local cams only. No-ops on backends that ignore them.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # 1-frame buffer keeps latency low for both USB and network streams.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_args():
    ap = argparse.ArgumentParser(description="Body and face tracker")
    ap.add_argument("--model", choices=("lite", "full"),
                    default="lite" if on_arm_board() else "full",
                    help="pose model (default: lite on ARM boards, else full)")
    ap.add_argument("--camera", type=int_or_str, default=0,
                    help="camera index (0,1,...) or RTSP/HTTP URL")
    ap.add_argument("--width", type=int, default=640, help="capture width")
    ap.add_argument("--height", type=int, default=480, help="capture height")
    ap.add_argument("--num-poses", type=int, default=1,
                    help="max people to track (default 1; higher costs more)")
    ap.add_argument("--mirror", action="store_true",
                    help="mirror the image (selfie view); off by default so "
                         "the view matches real life and gimbal yaw is correct")
    ap.add_argument("--headless", action="store_true",
                    help="run with no display window (onboard / over SSH)")
    ap.add_argument("--mavlink", metavar="CONN", default=None,
                    help="ArduPilot connection, e.g. udpout:127.0.0.1:14550 "
                         "or /dev/ttyAMA0 (aims the camera gimbal at target)")
    ap.add_argument("--baud", type=int, default=57600,
                    help="serial baud for --mavlink (default 57600)")
    ap.add_argument("--hfov", type=float, default=62.2,
                    help="camera horizontal field of view in degrees "
                         "(default 62.2 = Pi Camera v2)")
    ap.add_argument("--shoulder-width", type=float, default=0.40,
                    help="average shoulder width in metres, used to estimate "
                         "how far a person is (default 0.40)")
    ap.add_argument("--units", choices=("metric", "imperial"), default="metric",
                    help="distance display units (imperial = feet and inches)")
    ap.add_argument("--target-width", type=float, default=1.8,
                    help="real width in metres of a clicked object, for its "
                         "distance estimate (default 1.8 = a car)")
    ap.add_argument("--tracker", choices=("CSRT", "KCF", "DROTRACK"),
                    default="CSRT",
                    help="click-to-follow tracker: CSRT (accurate), KCF "
                         "(faster), or DROTRACK (drone-tuned, TF-free)")
    ap.add_argument("--detector", nargs="?", const="auto",
                    default=None, metavar="ONNX",
                    help="enable detection-assisted follow with a YOLOv8 ONNX "
                         "model (bare flag auto-picks the best bundled model: "
                         "yolov8m if present, else yolov8n). Snaps the click "
                         "onto the detected box and re-locks the tracker to "
                         "fresh detections, reporting a real loss when the "
                         "object is gone instead of drifting")
    ap.add_argument("--detect-classes", default=None, metavar="LIST",
                    help="comma-separated class names or ids to allow as "
                         "targets, e.g. car,van,truck,bus (default: any)")
    ap.add_argument("--detect-interval", type=int, default=15,
                    help="frames between detector re-locks (default 15)")
    ap.add_argument("--detect-conf", type=float, default=0.25,
                    help="detector confidence threshold (default 0.25)")
    ap.add_argument("--detect-tiles", default=None, metavar="CxR",
                    help="run detection on an overlapping CxR tile grid (e.g. "
                         "2x3) to recover small objects in high-res/4K frames; "
                         "slower (~tiles+1 inferences per frame)")
    # advanced flight: orbit-follow the target (opt-in, needs --mavlink)
    ap.add_argument("--follow", action="store_true",
                    help="ADVANCED: command the plane to ORBIT the tracked "
                         "target (GUIDED + armed only; never arms/changes mode)")
    ap.add_argument("--orbit-radius", type=float, default=80.0,
                    help="follow: standoff orbit radius in metres (default 80)")
    ap.add_argument("--orbit-speed", type=float, default=15.0,
                    help="follow: orbit ground speed in m/s (default 15)")
    ap.add_argument("--min-alt", type=float, default=30.0,
                    help="follow: refuse to command below this AGL m (default 30)")
    ap.add_argument("--geofence", type=float, default=300.0,
                    help="follow: max target distance from home m (default 300)")
    ap.add_argument("--cam-tilt", type=float, default=0.0,
                    help="follow: fixed camera down-tilt in degrees (default 0)")
    return ap.parse_args()


def main():
    args = parse_args()

    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODELS[args.model]),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max(1, args.num_poses),
            min_pose_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    grabber = FrameGrabber(open_camera(args)).start()

    # optional ArduPilot / UAV link
    uav = None
    if args.mavlink:
        from uav import MavlinkUAV, FollowConfig
        follow_cfg = FollowConfig(
            orbit_radius=args.orbit_radius, orbit_speed=args.orbit_speed,
            min_alt=args.min_alt, geofence_radius=args.geofence,
            cam_tilt_deg=args.cam_tilt)
        print(f"MAVLink: connecting to {args.mavlink} ...")
        uav = MavlinkUAV(args.mavlink, hfov_deg=args.hfov, baud=args.baud,
                         follow_config=follow_cfg)
        if uav.wait_heartbeat(timeout=10.0):
            mode = "ORBIT-FOLLOW" if args.follow else "gimbal aim"
            print(f"MAVLink: heartbeat received, mode: {mode}")
            uav.request_streams(rate_hz=10)
        else:
            print("MAVLink: no heartbeat (check link) - continuing anyway")
    if args.follow and not uav:
        raise SystemExit("--follow requires --mavlink")

    # optional ONNX detector for detection-assisted, drift-proof following
    detector = None
    target_classes = None
    if args.detector:
        from detector import build_detector, default_model_path, parse_tiles
        model_path = default_model_path(MODEL_DIR) if args.detector == "auto" \
            else args.detector
        print(f"Detector: loading {model_path} ...")
        detector = build_detector(model_path, conf=args.detect_conf,
                                  tiles=parse_tiles(args.detect_tiles))
        if args.detect_classes:
            ids = []
            for tok in args.detect_classes.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                cid = int(tok) if tok.isdigit() else detector.class_id(tok)
                if cid is None:
                    print(f"  warning: unknown class {tok!r}, ignoring")
                else:
                    ids.append(cid)
            target_classes = set(ids) or None
        names = [detector.names.get(c, c) for c in (target_classes or [])]
        print(f"Detector: ready ({len(detector.names)} classes)"
              + (f", targets: {', '.join(map(str, names))}" if names else ""))

    smoother = PointSmoother()
    assigner = TrackAssigner()
    follower = ObjectFollower(algo=args.tracker, detector=detector,
                              detect_interval=args.detect_interval,
                              target_classes=target_classes)
    obj_filter = None            # One-Euro pair for the object center
    hfov_rad = math.radians(args.hfov)
    imperial = args.units == "imperial"
    start = time.monotonic()
    fps = 0.0
    last_t = start
    last_seq = -1
    last_report = start

    # click-to-follow: left click picks an object (default box), left-drag
    # draws a box, right click clears. Only in windowed mode.
    WINDOW = "Body & Face Tracker  (q to quit)"
    mouse = {}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = (x, y)
            mouse["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse.get("down"):
            mouse["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and mouse.get("down"):
            x0, y0 = mouse.pop("down")
            if abs(x - x0) < 8 and abs(y - y0) < 8:
                mouse["click"] = (x, y)
            else:
                mouse["box"] = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse["clear"] = True

    if not args.headless:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while not grabber.stopped:
            frame, seq = grabber.read()
            if frame is None or seq == last_seq:
                time.sleep(0.001)  # no new frame yet; don't burn the CPU
                if not args.headless and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue
            last_seq = seq

            if args.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            screen_center = (w / 2, h / 2)

            now = time.monotonic()
            t = now - start
            timestamp_ms = int(t * 1000)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            # one pose -> up to two crosses: a FACE point and a BODY point,
            # plus an estimated real-world distance to that person (metres)
            dets = []  # (x, y, label, dist_m)
            for landmarks in pose_result.pose_landmarks:
                est = distance.estimate_distance(landmarks, w, h, hfov_rad,
                                                 shoulder_m=args.shoulder_width)
                dist_m = est[0] if est else None
                face = weighted_center(landmarks, FACE_LANDMARKS, w, h)
                if face is not None:
                    dets.append((face[0], face[1], "FACE", dist_m))
                body = weighted_center(landmarks, TORSO_LANDMARKS, w, h)
                if body is not None:
                    dets.append((body[0], body[1], "BODY", dist_m))

            # --- stable ids, smooth positions, distance to screen center ---
            keys = assigner.assign(dets, w, h)
            smoother.forget_missing(set(keys))
            tracked = []  # (x, y, label, dist_px, dist_m)
            for key, (dx, dy, label, dist_m) in zip(keys, dets):
                x, y = smoother.update(key, t, dx, dy)
                dist_px = math.hypot(x - screen_center[0], y - screen_center[1])
                tracked.append((x, y, label, dist_px, dist_m))

            # by default the GREEN target is the person closest to center
            closest_idx = min(range(len(tracked)), key=lambda i: tracked[i][3]) \
                if tracked else None

            # --- click-to-follow object (car etc.) overrides the person pick ---
            if mouse.pop("clear", False):
                follower.clear()
                obj_filter = None
            boxsel = mouse.pop("box", None)
            click = mouse.pop("click", None)
            if boxsel is not None:
                follower.start(frame, boxsel[0] + boxsel[2] / 2,
                               boxsel[1] + boxsel[3] / 2, box=boxsel)
                obj_filter = (OneEuroFilter(), OneEuroFilter())
            elif click is not None:
                follower.start(frame, click[0], click[1])
                obj_filter = (OneEuroFilter(), OneEuroFilter())

            obj_center = follower.update(frame, t) if follower.active else None
            obj_m = None
            obj_mph = None
            obj_size = None
            if obj_center is not None:
                ox, oy = obj_center
                if obj_filter:
                    ox, oy = obj_filter[0](t, ox), obj_filter[1](t, oy)
                obj_center = (ox, oy)
                bw_px = follower.box[2] if follower.box else 0
                if bw_px > 1:
                    obj_m = args.target_width * distance.focal_px(w, hfov_rad) / bw_px
                _, obj_mph = real_speed(follower.speed_px, obj_m,
                                        distance.focal_px(w, hfov_rad))
                nm = detector.names.get(follower.cls) if (
                    detector is not None and follower.cls is not None) else None
                if follower.box:
                    dims, measured = size.object_size(
                        nm, follower.box[2], follower.box[3],
                        follower.scene_scale)
                    if dims:
                        obj_size = (dims, measured)

            # unified target: clicked object if present, else the closest person
            if obj_center is not None:
                target_xy, target_m = obj_center, obj_m
                closest_idx = None          # people drawn red; object is green
            elif closest_idx is not None:
                target_xy = (tracked[closest_idx][0], tracked[closest_idx][1])
                target_m = tracked[closest_idx][4]
            else:
                target_xy, target_m = None, None
            target_px = math.hypot(target_xy[0] - screen_center[0],
                                   target_xy[1] - screen_center[1]) \
                if target_xy else None

            # --- UAV: aim the gimbal, and (if --follow) orbit the target ---
            follow_status = None
            if uav is not None:
                uav.update_from_telemetry()
                if target_xy is not None:
                    uav.send_gimbal(target_xy, screen_center, (w, h))
                    if args.follow:
                        follow_status = uav.follow_target(
                            target_xy, screen_center, (w, h), range_m=target_m)
                else:
                    uav.notify_no_target()

            if not args.headless:
                for i, (x, y, label, dist_px, dist_m) in enumerate(tracked):
                    is_closest = i == closest_idx
                    color = GREEN if is_closest else RED
                    draw_distance(frame, screen_center, (x, y))
                    draw_cross(frame, x, y, color)
                    tag = f"{label} d={dist_px:.0f}px"
                    if dist_m is not None:
                        tag += f" ~{distance.fmt_distance(dist_m, imperial)}"
                    if is_closest:
                        tag += " CLOSEST"
                    cv2.putText(frame, tag, (int(x) + CROSS_SIZE + 4, int(y) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                cv2.LINE_AA)

                # clicked object: green box + cross (the follow TARGET)
                if obj_center is not None and follower.box is not None:
                    bx, by, bw_, bh_ = follower.box
                    cv2.rectangle(frame, (bx, by), (bx + bw_, by + bh_), GREEN, 2)
                    draw_distance(frame, screen_center, obj_center)
                    draw_cross(frame, obj_center[0], obj_center[1], GREEN)
                    otag = f"TARGET d={target_px:.0f}px"
                    if obj_m is not None:
                        otag += f" ~{distance.fmt_distance(obj_m, imperial)}"
                    if obj_mph is not None:
                        otag += f"  {obj_mph:.0f} mph"
                    elif follower.speed_px > 1:
                        otag += f"  {follower.speed_px:.0f} px/s"
                    if obj_size is not None:
                        dims, measured = obj_size
                        otag += f"  {'' if measured else '~'}" \
                                f"{size.fmt_size(dims, imperial)}"
                    cv2.putText(frame, otag, (bx, by - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
                elif follower.active:   # temporarily lost, show a hint
                    cv2.putText(frame, "target lost...", (8, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

                # live drag rectangle while selecting
                if mouse.get("down") and mouse.get("cur"):
                    x0, y0 = mouse["down"]
                    x1, y1 = mouse["cur"]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), WHITE, 1)

                draw_cross(frame, *screen_center, BLUE, size=30)
                hud = f"{fps:.0f} fps  [{args.model}]"
                if target_px is not None:
                    lbl = "target" if obj_center is not None else "closest"
                    hud += f"  {lbl}: {target_px:.0f}px"
                if target_m is not None:
                    hud += f"  ~{distance.fmt_distance(target_m, imperial)}"
                hud += "   [click: follow object  right-click: clear]"
                cv2.putText(frame, hud, (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
                if follow_status is not None:
                    cv2.putText(frame, f"FOLLOW: {follow_status}", (8, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1,
                                cv2.LINE_AA)

                cv2.imshow(WINDOW, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            elif now - last_report >= 2.0:  # headless: periodic status line
                last_report = now
                d = f"{target_px:.0f}px" if target_px is not None else "none"
                rng = f"  ~{distance.fmt_distance(target_m, imperial)}" \
                    if target_m is not None else ""
                spd = f"  {obj_mph:.0f}mph" if obj_mph is not None else ""
                extra = f"  follow:{follow_status}" if follow_status else ""
                print(f"{fps:4.1f} fps  people:{len(pose_result.pose_landmarks)}"
                      f"  closest:{d}{rng}{spd}{extra}")
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
        cv2.destroyAllWindows()
        pose_landmarker.close()
        if uav is not None:
            uav.close()


if __name__ == "__main__":
    main()
