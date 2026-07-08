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


class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012) for one scalar signal.

    Adapts its smoothing to speed: heavy smoothing when the value is
    nearly still (no jitter), light smoothing when it moves fast (no lag).
    """

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, t, x):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        # smoothed derivative (px/s)
        a_d = self._alpha(dt, self.d_cutoff)
        dx = (x - self.x_prev) / dt
        dx_s = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_s

        # cutoff rises with speed -> less smoothing when moving fast
        cutoff = self.min_cutoff + self.beta * abs(dx_s)
        a = self._alpha(dt, cutoff)
        self.x_prev = a * x + (1 - a) * self.x_prev
        return self.x_prev


class PointSmoother:
    """One-Euro-filtered (x, y) points, keyed by track name."""

    def __init__(self):
        self.filters = {}

    def update(self, key, t, x, y):
        if key not in self.filters:
            self.filters[key] = (OneEuroFilter(), OneEuroFilter())
        fx, fy = self.filters[key]
        return fx(t, x), fy(t, y)

    def forget_missing(self, live_keys):
        for key in list(self.filters):
            if key not in live_keys:
                del self.filters[key]


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

    smoother = PointSmoother()
    assigner = TrackAssigner()
    hfov_rad = math.radians(args.hfov)
    start = time.monotonic()
    fps = 0.0
    last_t = start
    last_seq = -1
    last_report = start

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

            # the cross closest to the blue center cross is the GREEN target
            closest_idx = min(range(len(tracked)), key=lambda i: tracked[i][3]) \
                if tracked else None
            closest_distance = tracked[closest_idx][3] \
                if closest_idx is not None else None
            closest_m = tracked[closest_idx][4] \
                if closest_idx is not None else None

            # --- UAV: aim the gimbal, and (if --follow) orbit the target ---
            follow_status = None
            if uav is not None:
                uav.update_from_telemetry()
                if closest_idx is not None:
                    cx, cy, *_ = tracked[closest_idx]
                    uav.send_gimbal((cx, cy), screen_center, (w, h))
                    if args.follow:
                        follow_status = uav.follow_target(
                            (cx, cy), screen_center, (w, h), range_m=closest_m)
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
                        tag += f" ~{dist_m:.1f}m"
                    if is_closest:
                        tag += " CLOSEST"
                    cv2.putText(frame, tag, (int(x) + CROSS_SIZE + 4, int(y) + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                cv2.LINE_AA)

                draw_cross(frame, *screen_center, BLUE, size=30)
                hud = f"{fps:.0f} fps  [{args.model}]"
                if closest_distance is not None:
                    hud += f"  closest: {closest_distance:.0f}px"
                if closest_m is not None:
                    hud += f"  ~{closest_m:.1f}m"
                cv2.putText(frame, hud, (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
                if follow_status is not None:
                    cv2.putText(frame, f"FOLLOW: {follow_status}", (8, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1,
                                cv2.LINE_AA)

                cv2.imshow("Body & Face Tracker  (q to quit)", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            elif now - last_report >= 2.0:  # headless: periodic status line
                last_report = now
                d = f"{closest_distance:.0f}px" if closest_distance else "none"
                rng = f"  ~{closest_m:.1f}m" if closest_m is not None else ""
                extra = f"  follow:{follow_status}" if follow_status else ""
                print(f"{fps:4.1f} fps  people:{len(pose_result.pose_landmarks)}"
                      f"  closest:{d}{rng}{extra}")
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
