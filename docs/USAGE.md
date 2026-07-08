# Usage

## Install

Desktop (dev/testing):
```bash
pip install opencv-python==4.12.0.88 mediapipe pymavlink flask
```

Raspberry Pi (64-bit Raspberry Pi OS required for MediaPipe):
```bash
pip install opencv-python mediapipe pymavlink flask
```

The MediaPipe model files live in [`models/`](../models) and are already
present — no download needed.

## Desktop tracker (CLI)

```bash
python3 tracker.py                       # default: full model on desktop
python3 tracker.py --model lite          # faster, for low-power boards
python3 tracker.py --camera 1            # pick a camera index
python3 tracker.py --camera rtsp://…     # network / IP camera
python3 tracker.py --num-poses 3         # track up to 3 people
python3 tracker.py --mirror              # selfie view (off by default)
python3 tracker.py --headless            # no window (onboard / SSH)
```

Overlay: **red** cross on each face/body, **green** on the person closest to
center, **blue** cross at screen center, white lines labeled with pixel
distance. Press `q`/`Esc` to quit (`Ctrl+C` in headless).

### With a UAV
```bash
# aim the gimbal at the target (camera only)
python3 tracker.py --mavlink udpout:127.0.0.1:14550
python3 tracker.py --mavlink /dev/serial0 --baud 921600 --hfov 102

# advanced flight: orbit-follow the target (see SAFETY.md)
python3 tracker.py --mavlink udpout:127.0.0.1:14550 --follow \
    --orbit-radius 80 --orbit-speed 15 --min-alt 30 --geofence 300 --cam-tilt 20
```

Key flags: `--hfov` (match your camera), `--cam-tilt` (fixed camera down-tilt
for follow), `--orbit-radius/-speed`, `--min-alt`, `--geofence`.

## Web control panel

```bash
python3 app.py                           # http://127.0.0.1:5000
python3 app.py --host 0.0.0.0 --port 8080  # reachable from other devices
```

- **Add camera:** dropdown + Detect for local cams, or paste an RTSP/HTTP URL.
- **Tabs:** one per camera; select which the controls act on. × removes it.
- **View modes:** Single, **Auto-best** (follows the camera with the best view),
  **Grid** (all cameras at once). The best camera gets a green **BEST** badge.
- **Selected-camera controls:** model, people count, mirror.
- **UAV:** connect MAVLink and toggle Follow (per camera).
- **Outputs:** live video, fps/people/closest/mavlink stats, event log.

## Tests (no hardware needed)

```bash
python3 test_uav.py        # gimbal angle math over UDP loopback
python3 test_flight.py     # geo-projection + follow safety gates
```

## ArduPilot SITL (simulated flight)

```bash
# in an ArduPilot checkout
sim_vehicle.py -v ArduPlane --console --map
# arm, take off, switch to GUIDED, then on this repo:
python3 tracker.py --mavlink udpout:127.0.0.1:14550 --follow
```
Watch the plane orbit the tracked target in Mission Planner / the SITL map.
**Always validate follow in SITL before real flight.**

## Custom firmware

```bash
cd ardupilot_build
./build.sh --board CubeOrange        # clone + configure, stops before compile
./build.sh --board CubeOrange --full # actually compile (needs ARM toolchain)
./build.sh --sitl                    # configure a SITL build (host gcc)
```
See [`ardupilot_build/README.md`](../ardupilot_build/README.md).

## Hardware

Not bought yet — see [PARTS.md](PARTS.md) for a recommended parts list and how
each choice maps to these flags.
