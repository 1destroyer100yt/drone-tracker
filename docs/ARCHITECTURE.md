# Architecture

How the pieces fit together. The system is a body/face tracker that can drive a
camera gimbal and, optionally, an ArduPilot fixed-wing.

## Components

| File | Responsibility |
|------|----------------|
| [`tracker.py`](../tracker.py) | Desktop/CLI tracker: camera capture, MediaPipe pose, crosses, distance, optional MAVLink. |
| [`app.py`](../app.py) | Flask web control panel: multi-camera, RTSP, auto-best view, per-camera controls. Reuses `tracker.py`'s engine pieces. |
| [`uav.py`](../uav.py) | MAVLink link: gimbal aim (`MOUNT_CONTROL`) and orbit-follow (`DO_REPOSITION`) with safety gates. |
| [`geo.py`](../geo.py) | Pure geometry: camera line-of-sight → ground lat/lon (flat-ground or measured-range). Unit-tested. |
| [`distance.py`](../distance.py) | Monocular person-distance estimate (pinhole model via HFOV). Unit-tested. |
| [`test_uav.py`](../test_uav.py) | Gimbal angle tests (UDP loopback, no hardware). |
| [`test_flight.py`](../test_flight.py) | Geo-projection + follow safety-gate tests (fake MAVLink). |
| [`models/`](../models) | MediaPipe `.task` / `.tflite` model files (pose lite/full, face). |
| [`ardupilot_build/`](../ardupilot_build) | Custom ArduPilot Plane firmware config (features + build script). |

## Data flow

```
 camera (index / RTSP)
        │
        ▼
 FrameGrabber ── newest frame, drops stale ──► BGR→RGB
        │
        ▼
 MediaPipe PoseLandmarker (lite|full, VIDEO mode)
        │  face pts (nose/eyes/ears) + torso pts, visibility-weighted
        ▼
 One-Euro filter (per point)  ──►  crosses + distances to screen center
        │                                   │
        │                          closest = GREEN target
        ▼                                   ▼
 draw overlay (red/green/blue)      uav.send_gimbal()  → MOUNT_CONTROL
        │                           uav.follow_target()→ DO_REPOSITION (if --follow)
        ▼
 window / MJPEG stream (web) / headless log
```

## The tracking pipeline (accuracy choices)

- **One neural net.** Face and body both come from the pose model; there is no
  separate face detector, halving per-frame inference (important on a Pi).
- **Visibility weighting.** Face and torso centers weight each landmark by the
  model's confidence, so half-occluded points don't drag the cross.
- **One-Euro filter.** Smooths jitter when still, tracks fast motion with almost
  no lag — better than a moving average.
- **Stable identity.** `TrackAssigner` matches each detection to the nearest one
  from the previous frame, so the per-track smoothing filters don't swap between
  people when more than one is in view.
- **Click-to-follow.** `ObjectFollower` wraps an OpenCV CSRT tracker: click any
  object (a car, a bag) in the desktop window or the web video and it becomes the
  green TARGET, overriding the closest-person pick and driving the gimbal/follow.
  Works on arbitrary objects, not just people.
- **Distance estimate.** `distance.py` reads a body segment of known real size
  (shoulders ≈ 0.40 m, eyes ≈ 0.063 m) and the camera focal length (from
  `--hfov`) to estimate metres to each person. Feeds the on-screen readout and,
  for follow, a better target range than the flat-ground assumption.
- **Threaded capture.** A background grabber always hands over the newest frame
  and drops stale ones, keeping latency at ~one frame (matters for gimbal aim).

## Two tiers of aircraft interaction

1. **Gimbal aim (default with `--mavlink`).** Converts the target's pixel offset
   to pitch/yaw via the camera FOV and sends `MOUNT_CONTROL`. Camera moves; the
   aircraft does not.
2. **Orbit-follow (`--follow`).** Estimates the target's ground position from the
   camera line-of-sight + live telemetry ([`geo.py`](../geo.py)) and commands the
   plane to loiter around it at a standoff radius. Fenced by safety gates — see
   [SAFETY.md](SAFETY.md).

## Web app (multi-camera)

`app.py` runs one `TrackerEngine` per source (webcam index or RTSP URL), each in
its own thread with its own MJPEG stream and settings. An `EngineManager` scores
each camera's "view quality" every frame and picks a **best** camera (with
hysteresis) to drive the Auto-best view mode. MAVLink/follow is per-camera.

## Custom firmware

[`ardupilot_build/`](../ardupilot_build) reproduces what custom.ardupilot.org
does: an `extra_hwdef.dat` enabling mount/gimbal, camera, follow, and Lua
scripting, applied via `waf configure --extra-hwdef=…`. See its
[README](../ardupilot_build/README.md).
