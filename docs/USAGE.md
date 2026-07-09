# Usage

## Everything on one laptop (simplest)

The whole system — tracking, click-to-follow, distance, the web UI, and the
MAVLink/Follow command link — runs on a single laptop. Nothing needs to run
onboard unless you want the lowest-latency flight loop.

```bash
pip install -r requirements.txt
python3 app.py            # open http://127.0.0.1:5000
```

Then in the browser: **Add camera** (your built-in/USB webcam by index, or paste
the drone's **RTSP** URL), press **Start**, and **click the video** to follow an
object. To send commands to an aircraft, enter a **MAVLink connection** (your
SiK radio's serial port, or `udpout:127.0.0.1:14550` for ArduPilot SITL running
on the same laptop) and toggle Follow. The rest of this doc covers the CLI and
every option.

## Install

Desktop (dev/testing):
```bash
pip install -r requirements.txt
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
distance, and a **`~N.Nm` real-world distance** estimate per person. Press
`q`/`Esc` to quit (`Ctrl+C` in headless).

**Click to follow an object (e.g. a car):** in the desktop window, **left-click**
an object (or **left-drag** a box around it) to lock a CSRT tracker onto it — it
becomes the green TARGET and drives the gimbal/follow instead of the closest
person. **Right-click** to clear. Set the object's real width for its distance
estimate with `--target-width 1.8` (metres; 1.8 ≈ a car). Choose the tracker
with `--tracker CSRT` (accurate, default), `--tracker KCF` (faster), or
`--tracker DROTRACK` (a vendored, TensorFlow-free build of DroTrack, tuned for
drone footage — see [third_party/drotrack/NOTICE.md](../third_party/drotrack/NOTICE.md)).
In the web panel, **click the video** and use the **Tracker** / **Clear target**
buttons.

Compare the trackers on a synthetic sequence with `python3 benchmark_trackers.py`.
On clean synthetic video CSRT is the most accurate and DroTrack the fastest;
DroTrack's advantage is real aerial footage with camera ego-motion, so benchmark
on your own clips before choosing.

**Detection-assisted follow (recommended for drone footage):** add `--detector`
to back the click-tracker with our VisDrone-trained YOLOv8 model, run via ONNX
Runtime (no PyTorch needed). The bare flag auto-picks the best bundled model —
**yolov8m** (`models/visdrone_m.onnx`) if present, else the lighter **yolov8n**:

```bash
python3 tracker.py --detector                       # auto: yolov8m if present, else nano
python3 tracker.py --detector --detect-classes car,van,truck,bus
python3 tracker.py --detector models/visdrone_n.onnx # force the fast nano
```

With a detector the click **snaps onto the detected object box** (not a fixed
square), the tracker **re-locks to a fresh detection every `--detect-interval`
frames** so it can't slowly drift onto the background, and — crucially — when the
object truly leaves the frame the follower **reports the target LOST** instead of
confidently tracking nothing. That's the fix for the silent-drift failure the
plain trackers show on real aerial clips. Restrict targets with
`--detect-classes` (names or ids: pedestrian, people, bicycle, car, van, truck,
tricycle, awning-tricycle, bus, motor). Install the optional dep with
`pip install onnxruntime`.

**Model choice / speed:** yolov8m is much more accurate (mAP@50 0.42 vs the
nano's 0.30) but heavier — roughly **290 ms per detection on an M2 CPU** vs
**~40 ms** for yolov8n. The detector only runs every `--detect-interval` frames
(CSRT fills the gaps), so on CPU raise the interval for yolov8m
(e.g. `--detect-interval 30`) to stay smooth, or use the nano.

**Apple Neural Engine (real-time yolov8m on a Mac):** pass the CoreML package to
run on the Neural Engine — `--detector models/visdrone_m.mlpackage` — which is
**~43 ms/detect (~23/s), 6× faster** than ONNX-CPU (measured on an M2), i.e.
real-time. On a Mac the bare `--detector` flag picks this automatically when the
`.mlpackage` is present. Install the dep with `pip install coremltools pillow`
(CoreML is macOS-only; on a Pi/Linux use the ONNX models). Parity with the ONNX
model is exact (matching boxes at IoU ≥ 0.7).

**Distance:** estimated from shoulder width via the pinhole model; tune with
`--shoulder-width 0.40` (metres) if your subjects differ. It's a ±15-20%
estimate — see [SAFETY.md](SAFETY.md). Show it in **feet and inches** with
`--units imperial` (default `metric`); the web panel has a **Units: m/ft**
toggle. Internals stay metric — only the display changes.

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

## Where to run it — laptop vs onboard

Yes, you can run everything on your laptop and send commands to the aircraft.
There are two setups; pick by how tight the control loop needs to be.

**A. Laptop as ground station (easy, higher latency).**
The aircraft streams its camera (an onboard Pi/IP camera over **RTSP**), and your
laptop runs the tracker on that stream and sends commands back over a telemetry
radio.
```bash
python3 app.py                 # laptop; add the aircraft's RTSP URL as a camera
# connect MAVLink to your SiK radio on the laptop, e.g.:
#   /dev/tty.usbserial-XXXX   (macOS)   or   COM5 (Windows)   or
#   udpin:0.0.0.0:14550       (if a GCS like Mission Planner forwards MAVLink)
```
Click a car in the browser and it aims the gimbal / orbits it. Good for testing,
monitoring, and gimbal aim. The video-stream + radio round-trip adds latency, so
it's not ideal for tight, fast follow.

**B. Onboard companion (best for real follow, lowest latency).**
A Raspberry Pi on the aircraft runs the tracker headless with the local camera
and a direct serial link to the flight controller; your laptop just monitors.
```bash
# on the Pi (see PARTS.md for wiring)
python3 tracker.py --headless --mavlink /dev/serial0 --baud 921600 --follow
```
The perception→command loop stays on the aircraft, so a dropped laptop/Wi-Fi
link never affects control.

**Sending commands from the laptop, either way:** the `--mavlink` connection
string is the command channel. To a SiK radio use its serial port; to a shared
GCS use `udpin:0.0.0.0:14550` (have Mission Planner/MAVProxy forward there). All
the [safety gates](SAFETY.md) still apply — the aircraft only obeys Follow when
armed and in GUIDED, and the pilot's RC always overrides.

## Web control panel

```bash
python3 app.py                           # http://127.0.0.1:5000 (open, localhost)
python3 app.py --host 0.0.0.0 --port 8080  # reachable from other devices
python3 app.py --host 0.0.0.0 --token mysecret   # require a token
```

**Access control:** on localhost the panel is open for convenience. When you
bind to a network address (`--host 0.0.0.0`) a **token is required** — pass
`--token`, or one is auto-generated and printed. The startup line prints the
full URL with `?t=…`; open that. The page loads freely but every API/video call
needs the token. **Do not enable Follow on an untrusted network** — the panel
can command the aircraft.

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
python3 test_flight.py     # geo-projection (flat + range) + follow safety gates
python3 test_distance.py   # monocular person-distance pinhole math
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
