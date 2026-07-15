"""Web control panel for the body & face tracker -- multi-camera.

A Flask app that can run the tracker on several sources at once: local webcams
(by index) and RTSP / HTTP network cameras (by URL). Each source gets its own
engine, live annotated video stream, controls, and status. Pick a source from
the dropdown or paste an RTSP URL to add it.

Run:
  python3 app.py                 # then open http://127.0.0.1:5000
  python3 app.py --host 0.0.0.0 --port 8080

All detection / filtering / drawing / MAVLink logic is imported from
tracker.py and uav.py; this file only adds the web + multi-engine layer.
"""

import argparse
import math
import os
import secrets
import threading
import time
from collections import deque
from types import SimpleNamespace

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

import distance
import size
import tracker as T
from motion import real_speed


class TrackerEngine:
    """Runs one camera's tracking loop in a background thread and exposes the
    latest annotated frame plus live status. Options are plain attributes the
    web layer flips at any time; the loop reads them each frame."""

    def __init__(self, eid, source, detector=None, target_classes=None,
                 detect_interval=15):
        self.eid = eid
        self.source = source          # int index or str URL
        self.detector = detector
        self.lock = threading.Lock()
        self.thread = None
        self.running = False

        self.width = 640
        self.height = 480
        self.model = "lite" if T.on_arm_board() else "full"
        self.mirror = False
        self.num_poses = 1
        self.follow = False
        self.units = "metric"        # or "imperial" (feet + inches)
        self.target_width = 1.8      # clicked-object real width (m), a car

        # click-to-follow object tracker (seeded from a browser click). With a
        # detector it snaps onto the detected object box, re-locks to fresh
        # detections, and reports a real loss instead of drifting onto the
        # background (confidence-gated loss reporting).
        self.follower = T.ObjectFollower(
            detector=detector, target_classes=target_classes,
            detect_interval=detect_interval, coast_seconds=15.0)
        self._obj_filter = None
        self._pending_click = None   # (nx, ny) normalized, set by the web layer
        self._clear_target = False
        self.has_target = False

        self.jpeg = None
        self.status = {"fps": 0.0, "people": 0, "closest": None,
                       "range_m": None, "target": False, "mavlink": "off",
                       "follow": None, "score": 0.0, "target_cls": None,
                       "speed_mph": None, "size": None, "coasting": None}
        self.events = deque(maxlen=40)
        self.uav = None

    def label(self):
        return f"Camera {self.source}" if isinstance(self.source, int) \
            else str(self.source)

    def set_target(self, nx, ny):
        """Seed the object follower at a normalized (0..1) click position."""
        self._pending_click = (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))

    def clear_target(self):
        self._clear_target = True

    def log(self, msg):
        self.events.appendleft(f"{time.strftime('%H:%M:%S')}  {msg}")

    # ----- lifecycle --------------------------------------------------------
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log("tracking started")

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.status.update(fps=0.0, people=0, closest=None, follow=None)
        self.log("tracking stopped")

    # ----- MAVLink ----------------------------------------------------------
    def connect_mavlink(self, conn):
        from uav import MavlinkUAV, FollowConfig
        try:
            self.uav = MavlinkUAV(conn, follow_config=FollowConfig())
            ok = self.uav.wait_heartbeat(timeout=4.0)
            self.uav.request_streams(rate_hz=10)
            self.status["mavlink"] = "connected" if ok else "no heartbeat"
            self.log(f"MAVLink {conn}: "
                     + ("heartbeat OK" if ok else "no heartbeat (anyway)"))
        except Exception as e:
            self.uav = None
            self.status["mavlink"] = "error"
            self.log(f"MAVLink error: {e}")

    def disconnect_mavlink(self):
        if self.uav:
            self.uav.close()
            self.uav = None
        self.follow = False
        self.status.update(mavlink="off", follow=None)
        self.log("MAVLink disconnected")

    # ----- the loop ---------------------------------------------------------
    def _build_landmarker(self):
        return T.vision.PoseLandmarker.create_from_options(
            T.vision.PoseLandmarkerOptions(
                base_options=T.BaseOptions(
                    model_asset_path=T.POSE_MODELS[self.model]),
                running_mode=T.vision.RunningMode.VIDEO,
                num_poses=max(1, self.num_poses),
                min_pose_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        )

    def _run(self):
        cam_args = SimpleNamespace(camera=self.source, width=self.width,
                                   height=self.height)
        try:
            grabber = T.FrameGrabber(T.open_camera(cam_args)).start()
        except SystemExit as e:
            self.log(str(e))
            self.running = False
            return

        smoother = T.PointSmoother()
        assigner = T.TrackAssigner()
        hfov_rad = math.radians(62.2)   # web default; matches Pi Cam v2
        landmarker = None
        built_key = None
        start = time.monotonic()
        fps = 0.0
        last_t = start
        last_seq = -1

        while self.running and not grabber.stopped:
            want = (self.model, self.num_poses)
            if want != built_key:
                if landmarker:
                    landmarker.close()
                landmarker = self._build_landmarker()
                built_key = want
                self.log(f"model={self.model}  people<= {self.num_poses}")

            frame, seq = grabber.read()
            if frame is None or seq == last_seq:
                time.sleep(0.004)
                continue
            last_seq = seq

            if self.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            center = (w / 2, h / 2)
            now = time.monotonic()
            t = now - start
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
            last_t = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = T.mp.Image(image_format=T.mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, int(t * 1000))

            dets = []          # (x, y, label, dist_m)
            best_vis = 0.0
            vis_idx = T.FACE_LANDMARKS + T.TORSO_LANDMARKS
            for lm in result.pose_landmarks:
                vis = sum(lm[j].visibility for j in vis_idx) / len(vis_idx)
                best_vis = max(best_vis, vis)
                est = distance.estimate_distance(lm, w, h, hfov_rad)
                dist_m = est[0] if est else None
                face = T.weighted_center(lm, T.FACE_LANDMARKS, w, h)
                if face:
                    dets.append((face[0], face[1], "FACE", dist_m))
                body = T.weighted_center(lm, T.TORSO_LANDMARKS, w, h)
                if body:
                    dets.append((body[0], body[1], "BODY", dist_m))

            keys = assigner.assign(dets, w, h)
            smoother.forget_missing(set(keys))
            tracked = []       # (x, y, label, dist_px, dist_m)
            for key, (dx, dy, label, dist_m) in zip(keys, dets):
                x, y = smoother.update(key, t, dx, dy)
                d = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
                tracked.append((x, y, label, d, dist_m))

            ci = min(range(len(tracked)), key=lambda i: tracked[i][3]) \
                if tracked else None
            closest_d = tracked[ci][3] if ci is not None else None

            # "best view" score: a clear, centred, confident subject scores
            # high; empty frames score 0. Drives auto camera-swapping.
            if tracked:
                diag = (w * w + h * h) ** 0.5
                closeness = max(0.0, 1.0 - closest_d / (0.5 * diag))
                score = round(100 * (0.5 * closeness + 0.5 * best_vis), 1)
            else:
                score = 0.0

            # --- click-to-follow object overrides the person pick ---
            if self._clear_target:
                self._clear_target = False
                self.follower.clear()
                self._obj_filter = None
            if self._pending_click is not None:
                nx, ny = self._pending_click
                self._pending_click = None
                self.follower.start(frame, nx * w, ny * h)
                self._obj_filter = (T.OneEuroFilter(), T.OneEuroFilter())
                self.log(f"target set at ({nx:.2f},{ny:.2f})")
            obj_center = self.follower.update(frame, t) \
                if self.follower.alive else None
            obj_m = None
            obj_mph = None
            obj_size = None
            if obj_center is not None:
                ox, oy = obj_center
                if self._obj_filter:
                    ox = self._obj_filter[0](t, ox)
                    oy = self._obj_filter[1](t, oy)
                obj_center = (ox, oy)
                bw_px = self.follower.box[2] if self.follower.box else 0
                if bw_px > 1:
                    obj_m = self.target_width * distance.focal_px(
                        w, hfov_rad) / bw_px
                _, obj_mph = real_speed(self.follower.speed_px, obj_m,
                                        distance.focal_px(w, hfov_rad))
                nm = self.detector.names.get(self.follower.cls) if (
                    self.detector is not None
                    and self.follower.cls is not None) else None
                if self.follower.box:
                    dims, meas = size.object_size(
                        nm, self.follower.box[2], self.follower.box[3],
                        self.follower.scene_scale)
                    if dims:
                        obj_size = ("" if meas else "~") + size.fmt_size(
                            dims, self.units == "imperial")
            self.has_target = obj_center is not None

            if obj_center is not None:
                target_xy, target_m = obj_center, obj_m
                ci = None                     # people red; object is green
            elif ci is not None:
                target_xy, target_m = (tracked[ci][0], tracked[ci][1]), \
                    tracked[ci][4]
            else:
                target_xy, target_m = None, None

            follow_status = None
            if self.uav is not None:
                self.uav.update_from_telemetry()
                if target_xy is not None:
                    self.uav.send_gimbal(target_xy, center, (w, h))
                    if self.follow:
                        follow_status = self.uav.follow_target(
                            target_xy, center, (w, h), range_m=target_m)
                else:
                    self.uav.notify_no_target()

            for i, (x, y, label, d, dist_m) in enumerate(tracked):
                is_c = i == ci
                color = T.GREEN if is_c else T.RED
                T.draw_distance(frame, center, (x, y))
                T.draw_cross(frame, x, y, color)
                tag = f"{label} d={d:.0f}px"
                if dist_m is not None:
                    tag += " ~" + distance.fmt_distance(
                        dist_m, self.units == "imperial")
                if is_c:
                    tag += " CLOSEST"
                cv2.putText(frame, tag, (int(x) + T.CROSS_SIZE + 4, int(y) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # clicked object: green box + cross (the follow TARGET)
            if obj_center is not None and self.follower.box is not None:
                bx, by, bw_, bh_ = self.follower.box
                cv2.rectangle(frame, (bx, by), (bx + bw_, by + bh_), T.GREEN, 2)
                T.draw_distance(frame, center, obj_center)
                T.draw_cross(frame, obj_center[0], obj_center[1], T.GREEN)
                otag = "TARGET"
                if self.detector is not None and self.follower.cls is not None:
                    nm = self.detector.names.get(self.follower.cls, "")
                    if nm:
                        otag += f" ({nm})"
                if obj_m is not None:
                    otag += " ~" + distance.fmt_distance(
                        obj_m, self.units == "imperial")
                if obj_mph is not None:
                    otag += f"  {obj_mph:.0f} mph"
                elif self.follower.speed_px > 1:
                    otag += f"  {self.follower.speed_px:.0f} px/s"
                if obj_size:
                    otag += "  " + obj_size
                cv2.putText(frame, otag, (bx, by - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, T.GREEN, 1, cv2.LINE_AA)
            elif self.follower.coasting:
                cv2.putText(frame,
                            f"COASTING {self.follower.coast_elapsed:.0f}s (re-id)",
                            (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 255), 1, cv2.LINE_AA)
            elif self.follower.active:
                cv2.putText(frame, "target lost...", (8, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, T.RED, 1, cv2.LINE_AA)
            T.draw_cross(frame, *center, T.BLUE, size=30)
            cv2.putText(frame, self.label(), (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, T.WHITE, 1, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()
            target_px = None
            if target_xy is not None:
                target_px = round(math.hypot(target_xy[0] - center[0],
                                             target_xy[1] - center[1]))
            target_cls = None
            if self.has_target and self.detector is not None \
                    and self.follower.cls is not None:
                target_cls = self.detector.names.get(self.follower.cls)
            self.status.update(
                fps=round(fps, 1), people=len(result.pose_landmarks),
                closest=target_px,
                range_m=None if target_m is None else round(target_m, 1),
                target=self.has_target, follow=follow_status, score=score,
                target_cls=target_cls, size=obj_size,
                speed_mph=None if obj_mph is None else round(obj_mph, 1),
                coasting=round(self.follower.coast_elapsed, 1)
                if self.follower.coasting else None)

        grabber.stop()
        if landmarker:
            landmarker.close()
        self.running = False

    def frame_bytes(self):
        with self.lock:
            return self.jpeg

    def snapshot(self):
        return dict(id=self.eid, source=self.source, label=self.label(),
                    running=self.running, model=self.model, mirror=self.mirror,
                    num_poses=self.num_poses, mavlink=self.status["mavlink"],
                    follow=self.follow, units=self.units,
                    target=self.has_target, tracker=self.follower.algo,
                    detector=self.detector is not None,
                    score=self.status["score"],
                    status=self.status, events=list(self.events))


class EngineManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.engines = {}      # eid -> TrackerEngine
        self._counter = 0
        self.detector = None            # shared YoloOnnxDetector (or None)
        self.target_classes = None      # allowed class ids for targets
        self.detect_interval = 15
        self.available = []    # last camera-detect result
        self._best = None      # current best-view camera id
        self._score_floor = 5.0    # below this, nobody is really in view
        self._switch_margin = 15.0  # candidate must beat current by this to win

    def add(self, source):
        with self.lock:
            self._counter += 1
            eid = f"cam{self._counter}"
            eng = TrackerEngine(eid, source, detector=self.detector,
                                target_classes=self.target_classes,
                                detect_interval=self.detect_interval)
            self.engines[eid] = eng
        eng.log(f"added source {source!r}")
        eng.start()
        return eid

    def remove(self, eid):
        with self.lock:
            eng = self.engines.pop(eid, None)
        if eng:
            eng.stop()
            eng.disconnect_mavlink()

    def get(self, eid):
        return self.engines.get(eid)

    def _update_best_locked(self):
        """Pick the running camera with the highest view score. Hysteresis:
        keep the current best until a rival clearly beats it, so the auto view
        doesn't flip back and forth on near-ties."""
        running = {eid: e for eid, e in self.engines.items() if e.running}
        if not running:
            self._best = None
            return None
        scores = {eid: e.status.get("score", 0.0) for eid, e in running.items()}
        top_id = max(scores, key=scores.get)
        top = scores[top_id]
        cur = self._best if self._best in running else None
        if top < self._score_floor:
            self._best = None                       # nobody meaningfully in view
        elif cur is None:
            self._best = top_id
        elif top_id != cur and top > scores[cur] + self._switch_margin:
            self._best = top_id                     # clear winner -> switch
        # otherwise keep the current best
        return self._best

    def snapshot(self):
        with self.lock:
            engines = [e.snapshot() for e in self.engines.values()]
            best = self._update_best_locked()
        return {"engines": engines, "available": self.available, "best": best}

    def detect_cameras(self, max_index=5):
        """Probe local camera indices not already in use; report which open."""
        in_use = {e.source for e in self.engines.values()
                  if isinstance(e.source, int)}
        found = []
        for idx in range(max_index + 1):
            if idx in in_use:
                found.append(idx)
                continue
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                found.append(idx)
            cap.release()
        self.available = found
        return found


manager = EngineManager()
app = Flask(__name__)


@app.before_request
def _require_token():
    """If a token is configured, protect the control/video endpoints. The page
    itself loads freely; it passes the token (from its own URL) on every call."""
    tok = app.config.get("TOKEN")
    if not tok:
        return
    if request.path.startswith("/api") or request.path.startswith("/video"):
        given = request.headers.get("X-Token") or request.args.get("t")
        if given != tok:
            return "unauthorized", 401


PAGE = r"""
<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Body &amp; Face Tracker</title>
<style>
 :root{--bg:#0f1216;--panel:#1a1f27;--line:#2a313c;--fg:#e6e9ee;
   --muted:#8b95a3;--accent:#3b82f6;--green:#22c55e;--red:#ef4444}
 *{box-sizing:border-box}
 body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
   background:var(--bg);color:var(--fg)}
 header{padding:16px 22px;border-bottom:1px solid var(--line);
   display:flex;align-items:center;gap:12px}
 header h1{font-size:17px;margin:0;font-weight:600}
 header .dot{width:9px;height:9px;border-radius:50%;background:var(--muted)}
 header .dot.on{background:var(--green)}
 header #hstatus{color:var(--muted);font-size:13px}
 main{display:grid;grid-template-columns:290px 1fr;gap:18px;padding:18px;
   max-width:1160px;margin:0 auto}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
   padding:14px;margin-bottom:14px}
 .card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;
   color:var(--muted);margin:0 0 10px}
 .row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
 .row:last-child{margin-bottom:0}
 button,select{font:inherit;color:var(--fg);background:#232a34;
   border:1px solid var(--line);border-radius:8px;padding:8px 12px;cursor:pointer;
   transition:.12s}
 button:hover{border-color:var(--accent)}
 button.on{background:var(--accent);border-color:var(--accent);color:#fff}
 button.primary{background:var(--green);border-color:var(--green);color:#06210f;
   font-weight:600}
 button.danger.on{background:var(--red);border-color:var(--red);color:#fff}
 button:disabled{opacity:.4;cursor:not-allowed}
 .seg button{flex:1}
 input[type=text]{font:inherit;background:#0e1319;color:var(--fg);
   border:1px solid var(--line);border-radius:8px;padding:8px;width:100%}
 label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px}
 .tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
 .tab{display:flex;align-items:center;gap:8px;background:var(--panel);
   border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer;
   font-size:13px;max-width:230px}
 .tab.sel{border-color:var(--accent)}
 .tab .x{color:var(--muted);border:0;background:0;padding:0 2px;font-size:15px}
 .tab .lbl{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .video{background:#000;border:1px solid var(--line);border-radius:10px;
   overflow:hidden;aspect-ratio:4/3;display:flex;align-items:center;
   justify-content:center}
 .video img{width:100%;height:100%;object-fit:contain}
 .empty{color:var(--muted);font-size:14px;padding:60px 0;text-align:center}
 .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(84px,1fr));
   gap:10px;margin-top:14px}
 .stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;
   padding:12px;text-align:center}
 .stat .v{font-size:22px;font-weight:600}
 .stat .k{font-size:11px;color:var(--muted);text-transform:uppercase;
   letter-spacing:.05em}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
   gap:10px;margin-top:14px}
 .grid figure{margin:0;border:1px solid var(--line);border-radius:8px;
   overflow:hidden;cursor:pointer;background:#000}
 .grid figure.sel{border-color:var(--accent)}
 .grid img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}
 .grid figcaption{font-size:11px;color:var(--muted);padding:4px 6px;
   background:var(--panel);overflow:hidden;text-overflow:ellipsis;
   white-space:nowrap}
 .viewbar{display:flex;align-items:center;gap:12px;margin-bottom:14px;
   font-size:12px;color:var(--muted);text-transform:uppercase;
   letter-spacing:.05em}
 .viewbar .seg{display:flex;gap:8px;flex:0 0 auto}
 .viewbar #bestlbl{text-transform:none;letter-spacing:0}
 .badge{font-size:10px;background:var(--green);color:#06210f;border-radius:5px;
   padding:1px 6px;font-weight:700;letter-spacing:0}
 .grid.big{grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
 #log{background:#0e1319;border:1px solid var(--line);border-radius:10px;
   padding:12px;margin-top:14px;font:12px/1.6 ui-monospace,Menlo,monospace;
   color:var(--muted);height:140px;overflow:auto;white-space:pre-wrap}
 @media(max-width:760px){main{grid-template-columns:1fr}}
</style></head><body>
<header><span class="dot" id="livedot"></span><h1>Body &amp; Face Tracker</h1>
 <span id="hstatus"></span></header>
<main>
 <div>
  <div class="card">
   <h2>Add camera</h2>
   <label>Local camera</label>
   <div class="row">
     <select id="camSel" style="flex:1"></select>
     <button onclick="detect()">Detect</button></div>
   <div class="row"><button style="flex:1" onclick="addLocal()">Add camera</button></div>
   <label>Network camera (RTSP / HTTP)</label>
   <div class="row"><input type=text id="rtsp"
     placeholder="rtsp://user:pass@192.168.1.50:554/stream"></div>
   <div class="row"><button style="flex:1" onclick="addRtsp()">Add stream</button></div>
  </div>
  <div class="card">
   <h2>Selected camera</h2>
   <div class="row"><button class="primary" id="startBtn" style="flex:1"
     onclick="toggleRun()" disabled>Start</button></div>
   <label>Pose model</label>
   <div class="row seg">
     <button data-model="lite" onclick="setModel('lite')">Lite</button>
     <button data-model="full" onclick="setModel('full')">Full</button></div>
   <label>People to track</label>
   <div class="row seg">
     <button data-poses="1" onclick="setPoses(1)">1</button>
     <button data-poses="2" onclick="setPoses(2)">2</button>
     <button data-poses="3" onclick="setPoses(3)">3</button>
     <button data-poses="4" onclick="setPoses(4)">4</button></div>
   <div class="row">
     <button id="mirrorBtn" style="flex:1" onclick="toggleMirror()">Mirror: off</button>
     <button id="unitsBtn" style="flex:1" onclick="toggleUnits()">Units: m</button>
   </div>
  </div>
  <div class="card">
   <h2>UAV (ArduPilot)</h2>
   <label>MAVLink connection</label>
   <div class="row"><input type=text id="conn" value="udpout:127.0.0.1:14550"></div>
   <div class="row"><button id="mavBtn" style="flex:1" onclick="toggleMav()">Connect</button></div>
   <div class="row"><button class="danger" id="followBtn" style="flex:1"
     onclick="toggleFollow()" disabled>Follow: off</button></div>
   <p style="font-size:11px;color:var(--muted);margin:4px 0 0">Follow orbits the
     target — GUIDED + armed only. Gimbal aim is automatic when connected.</p>
  </div>
 </div>
 <div>
  <div class="viewbar">
    <span>View</span>
    <div class="seg">
      <button data-view="single" onclick="setView('single')">Single</button>
      <button data-view="auto" onclick="setView('auto')">Auto-best</button>
      <button data-view="grid" onclick="setView('grid')">Grid</button>
    </div>
    <span id="bestlbl"></span>
  </div>
  <div class="tabs" id="tabs"></div>
  <div class="video" id="videoWrap">
    <div class="empty" id="empty">Add a camera to begin.</div>
    <img id="vid" alt="video" style="display:none;cursor:crosshair"
         onclick="vidClick(event)">
  </div>
  <p style="font-size:12px;color:var(--muted);margin:8px 0 0">
    Click the video to follow an object (e.g. a car).
    <button id="trackerBtn" style="padding:4px 10px;margin-left:8px"
            onclick="toggleTracker()">Tracker: CSRT</button>
    <button id="clrTargetBtn" style="padding:4px 10px;margin-left:4px"
            onclick="clearTarget()">Clear target</button></p>
  <div class="stats">
   <div class="stat"><div class="v" id="s_fps">–</div><div class="k">fps</div></div>
   <div class="stat"><div class="v" id="s_people">–</div><div class="k">people</div></div>
   <div class="stat"><div class="v" id="s_closest">–</div><div class="k">closest px</div></div>
   <div class="stat"><div class="v" id="s_range">–</div><div class="k">distance</div></div>
   <div class="stat"><div class="v" id="s_mav">off</div><div class="k">mavlink</div></div>
  </div>
  <div class="grid" id="grid"></div>
  <div id="log"></div>
 </div>
</main>
<script>
let engines=[], sel=null, available=[], best=null, viewMode='single';
const TOKEN=new URLSearchParams(location.search).get('t')||'';
function q(u){return TOKEN?u+(u.includes('?')?'&':'?')+'t='+encodeURIComponent(TOKEN):u;}
function setView(m){viewMode=m; if(m==='auto'&&best)sel=best; render();}
async function ctl(action,value){
  await fetch(q('/api/control'),{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:sel,action,value})}); poll();}
async function detect(){
  const r=await (await fetch(q('/api/cameras/detect'))).json();
  available=r.available; fillCamSel();}
function fillCamSel(){
  const s=document.getElementById('camSel');
  s.innerHTML = (available.length?available:[0,1,2,3])
    .map(i=>`<option value="${i}">Camera ${i}</option>`).join('');}
async function addLocal(){
  const v=+document.getElementById('camSel').value;
  const r=await (await fetch(q('/api/cameras/add'),{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source:v})})).json();
  sel=r.id; poll();}
async function addRtsp(){
  const u=document.getElementById('rtsp').value.trim(); if(!u)return;
  const r=await (await fetch(q('/api/cameras/add'),{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source:u})})).json();
  sel=r.id; document.getElementById('rtsp').value=''; poll();}
async function removeCam(id,ev){ev.stopPropagation();
  await fetch(q('/api/cameras/remove'),{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id})});
  if(sel===id)sel=null; poll();}
function pick(id){sel=id; render();}
function toggleRun(){const e=cur(); if(e)ctl(e.running?'stop':'start');}
function setModel(m){ctl('set_model',m);}
function setPoses(n){ctl('set_poses',n);}
function toggleMirror(){const e=cur(); if(e)ctl('set_mirror',!e.mirror);}
function toggleUnits(){const e=cur(); if(e)ctl('set_units',e.units==='imperial'?'metric':'imperial');}
function fmtM(m,imperial){
  if(m==null)return '–';
  if(!imperial)return m.toFixed(1)+'m';
  let ti=m*39.37007874, ft=Math.floor(ti/12), inch=Math.round(ti-ft*12);
  if(inch===12){ft++;inch=0;} return ft+'ft '+inch+'in';}
function toggleMav(){const e=cur(); if(!e)return;
  const c=(e.mavlink==='connected'||e.mavlink==='no heartbeat');
  ctl(c?'disconnect_mavlink':'connect_mavlink',
      document.getElementById('conn').value);}
function toggleFollow(){const e=cur(); if(e)ctl('set_follow',!e.follow);}
function clearTarget(){const e=cur(); if(e)ctl('clear_target');}
function toggleTracker(){const e=cur(); if(!e)return;
  const nx={CSRT:'KCF',KCF:'DROTRACK',DROTRACK:'CSRT'}; ctl('set_tracker',nx[e.tracker]||'CSRT');}
function vidClick(ev){
  const e=cur(); if(!e)return;
  const img=ev.target, r=img.getBoundingClientRect();
  const nw=img.naturalWidth||r.width, nh=img.naturalHeight||r.height;
  const scale=Math.min(r.width/nw, r.height/nh);   // object-fit: contain
  const dw=nw*scale, dh=nh*scale, ox=(r.width-dw)/2, oy=(r.height-dh)/2;
  const x=(ev.clientX-r.left-ox)/dw, y=(ev.clientY-r.top-oy)/dh;
  if(x<0||x>1||y<0||y>1)return;                     // clicked the letterbox
  ctl('set_target',{x,y});}
function cur(){return engines.find(e=>e.id===sel)||null;}

function render(){
  document.querySelectorAll('[data-view]').forEach(b=>
    b.className=b.dataset.view===viewMode?'on':'');
  // tabs
  document.getElementById('tabs').innerHTML = engines.map(e=>
    `<div class="tab ${e.id===sel?'sel':''}" onclick="pick('${e.id}')">
      <span class="dot ${e.running?'on':''}"></span>
      <span class="lbl">${e.label}</span>
      ${e.id===best?'<span class="badge">BEST</span>':''}
      <button class="x" onclick="removeCam('${e.id}',event)">×</button></div>`
    ).join('');
  // multi-view grid (enlarged in Grid mode)
  document.getElementById('grid').className='grid'+(viewMode==='grid'?' big':'');
  document.getElementById('grid').innerHTML = engines.map(e=>
    `<figure class="${e.id===sel?'sel':''}" onclick="pick('${e.id}')">
      <img src="${q('/video/'+e.id)}"><figcaption>${e.id===best?'<span class="badge">BEST</span> ':''}${e.label} · ${e.status.fps} fps · view ${e.score}
      </figcaption></figure>`).join('');
  // in Grid mode the big single view is hidden; auto follows the best cam
  document.getElementById('videoWrap').style.display=
     viewMode==='grid'?'none':'';
  const bl=document.getElementById('bestlbl');
  if(viewMode==='auto'){
    const b=engines.find(x=>x.id===best);
    bl.textContent='Auto → '+(b?b.label:'(waiting for a subject)');
  }else bl.textContent='';
  const e=cur();
  const vid=document.getElementById('vid'), empty=document.getElementById('empty');
  if(e){
    empty.style.display='none'; vid.style.display='';
    if(vid.dataset.id!==e.id){vid.src=q('/video/'+e.id); vid.dataset.id=e.id;}
  }else{empty.style.display=''; vid.style.display='none'; vid.dataset.id='';}
  // selected-camera controls
  const has=!!e;
  const sb=document.getElementById('startBtn');
  sb.disabled=!has; sb.textContent=has&&e.running?'Stop':'Start';
  sb.className=has&&e.running?'danger on':'primary';
  document.getElementById('livedot').className='dot'+(has&&e.running?' on':'');
  document.getElementById('mirrorBtn').textContent='Mirror: '+(has&&e.mirror?'on':'off');
  document.getElementById('mirrorBtn').className=has&&e.mirror?'on':'';
  const imp=has&&e.units==='imperial';
  document.getElementById('unitsBtn').textContent='Units: '+(imp?'ft':'m');
  document.getElementById('unitsBtn').className=imp?'on':'';
  document.querySelectorAll('[data-model]').forEach(b=>
    b.className=has&&b.dataset.model===e.model?'on':'');
  document.querySelectorAll('[data-poses]').forEach(b=>
    b.className=has&&+b.dataset.poses===e.num_poses?'on':'');
  const conn=has&&(e.mavlink==='connected'||e.mavlink==='no heartbeat');
  const mb=document.getElementById('mavBtn');
  mb.textContent=conn?'Disconnect':'Connect'; mb.className=conn?'on':'';
  const fb=document.getElementById('followBtn');
  fb.disabled=!conn; fb.textContent='Follow: '+(has&&e.follow?'on':'off');
  fb.className='danger'+(has&&e.follow?' on':'');
  document.getElementById('s_fps').textContent=has?e.status.fps:'–';
  document.getElementById('s_people').textContent=has?e.status.people:'–';
  document.getElementById('s_closest').textContent=
     has&&e.status.closest!=null?e.status.closest:'–';
  document.getElementById('s_range').textContent=
     has?fmtM(e.status.range_m,imp):'–';
  document.getElementById('s_mav').textContent=has?e.mavlink:'off';
  const clr=document.getElementById('clrTargetBtn');
  clr.className=(e&&e.status.target)?'on':''; clr.disabled=!(e&&e.status.target);
  document.getElementById('trackerBtn').textContent='Tracker: '+(e?e.tracker:'CSRT');
  let hs=`${engines.length} camera${engines.length!==1?'s':''}`;
  if(e&&e.status.target)hs+=' · TARGET LOCKED'+(e.status.target_cls?' ('+e.status.target_cls+')':'')+(e.status.speed_mph!=null?' · '+e.status.speed_mph+' mph':'')+(e.status.size?' · '+e.status.size:'');
  if(e&&e.status.coasting!=null)hs+=' · COASTING '+e.status.coasting+'s (re-id)';
  if(e&&e.status.follow)hs+=' · follow: '+e.status.follow;
  document.getElementById('hstatus').textContent=hs;
  document.getElementById('log').textContent=e?e.events.join('\n'):'';
}
async function poll(){
  try{const s=await (await fetch(q('/api/state'))).json();
    engines=s.engines; available=s.available; best=s.best;
    if(viewMode==='auto'&&best)sel=best;
    else if(!sel&&engines.length)sel=engines[0].id;
    render();}catch(err){}
}
fillCamSel(); poll(); setInterval(poll,700);
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video/<eid>")
def video(eid):
    eng = manager.get(eid)
    if eng is None:
        return "no such camera", 404

    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while manager.get(eid) is not None:
            buf = eng.frame_bytes()
            if buf is None:
                time.sleep(0.05)
                continue
            yield boundary + buf + b"\r\n"
            time.sleep(0.04)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def state():
    return jsonify(manager.snapshot())


@app.route("/api/cameras/detect")
def detect():
    return jsonify(available=manager.detect_cameras())


@app.route("/api/cameras/add", methods=["POST"])
def add_camera():
    source = request.get_json(force=True).get("source")
    if isinstance(source, str):
        source = source.strip()
        if not source:
            return jsonify(ok=False, error="empty source"), 400
    eid = manager.add(source)
    return jsonify(ok=True, id=eid)


@app.route("/api/cameras/remove", methods=["POST"])
def remove_camera():
    manager.remove(request.get_json(force=True).get("id"))
    return jsonify(ok=True)


@app.route("/api/control", methods=["POST"])
def control():
    data = request.get_json(force=True)
    eng = manager.get(data.get("id"))
    if eng is None:
        return jsonify(ok=False, error="no such camera"), 404
    action, value = data.get("action"), data.get("value")
    if action == "start":
        eng.start()
    elif action == "stop":
        eng.stop()
    elif action == "set_model" and value in ("lite", "full"):
        eng.model = value
    elif action == "set_poses":
        eng.num_poses = max(1, min(4, int(value)))
    elif action == "set_mirror":
        eng.mirror = bool(value)
    elif action == "set_units" and value in ("metric", "imperial"):
        eng.units = value
    elif action == "set_target":
        try:
            eng.set_target(float(value["x"]), float(value["y"]))
        except (TypeError, KeyError, ValueError):
            return jsonify(ok=False, error="set_target needs {x,y}"), 400
    elif action == "clear_target":
        eng.clear_target()
    elif action == "set_tracker" and value in ("CSRT", "KCF", "DROTRACK"):
        eng.follower.algo = value       # applies on the next click/re-lock
        eng.log(f"object tracker: {value}")
    elif action == "connect_mavlink":
        eng.connect_mavlink(str(value))
    elif action == "disconnect_mavlink":
        eng.disconnect_mavlink()
    elif action == "set_follow":
        if eng.uav is None:
            eng.log("follow ignored: connect MAVLink first")
        else:
            eng.follow = bool(value)
            if eng.follow:
                eng.log("FOLLOW ENABLED via web - plane will orbit target; "
                        "keep manual override ready")
            else:
                eng.log("follow off")
    else:
        return jsonify(ok=False, error="unknown action"), 400
    return jsonify(ok=True)


def main():
    ap = argparse.ArgumentParser(description="Tracker web control panel")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--token", default=None,
                    help="require this access token on the API/video endpoints; "
                         "auto-generated when --host is not localhost")
    ap.add_argument("--detector", nargs="?", const="auto",
                    default=None, metavar="ONNX",
                    help="enable detection-assisted follow with a YOLOv8 ONNX "
                         "model (bare flag auto-picks the best bundled model: "
                         "yolov8m if present, else yolov8n); a click snaps onto "
                         "the detected object and re-locks to fresh detections")
    ap.add_argument("--detect-classes", default=None, metavar="LIST",
                    help="comma-separated class names/ids to allow as targets, "
                         "e.g. car,van,truck,bus (default: any)")
    ap.add_argument("--detect-interval", type=int, default=15,
                    help="frames between detector re-locks (default 15)")
    ap.add_argument("--detect-conf", type=float, default=0.25,
                    help="detector confidence threshold (default 0.25)")
    ap.add_argument("--detect-tiles", default=None, metavar="CxR",
                    help="run detection on an overlapping CxR tile grid (e.g. "
                         "2x3) to recover small objects in high-res/4K frames")
    args = ap.parse_args()

    if args.detector:
        from detector import build_detector, default_model_path, parse_tiles
        model_path = default_model_path(T.MODEL_DIR) if args.detector == "auto" \
            else args.detector
        det = build_detector(model_path, conf=args.detect_conf,
                             tiles=parse_tiles(args.detect_tiles))
        classes = None
        if args.detect_classes:
            ids = []
            for tok in args.detect_classes.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                cid = int(tok) if tok.isdigit() else det.class_id(tok)
                if cid is not None:
                    ids.append(cid)
            classes = set(ids) or None
        manager.detector = det
        manager.target_classes = classes
        manager.detect_interval = args.detect_interval
        names = [det.names.get(c, c) for c in sorted(classes)] if classes else []
        print(f"Detector: {args.detector} ({len(det.names)} classes)"
              + (f", targets: {', '.join(map(str, names))}" if names else ""))

    token = args.token or os.environ.get("APP_TOKEN")
    open_host = args.host not in ("127.0.0.1", "localhost", "::1")
    if open_host and not token:
        token = secrets.token_urlsafe(12)   # never expose an open panel unguarded
    app.config["TOKEN"] = token

    if open_host:
        print("WARNING: this panel is reachable on the network. Anyone with the")
        print("         URL + token can control cameras and the MAVLink / Follow")
        print("         link. Do not enable Follow on an untrusted network.")
    url = f"http://{args.host}:{args.port}/" + (f"?t={token}" if token else "")
    print(f"Tracker web UI on {url}")
    if token:
        print(f"access token: {token}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
