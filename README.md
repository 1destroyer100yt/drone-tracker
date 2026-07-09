# Body & Face Tracker — person-following camera UAV

A Python tracker that finds people with MediaPipe, marks each with a cross,
measures pixel distance to screen center, and can aim a camera gimbal at — or
fly a fixed-wing plane to orbit — the closest person. Runs on a laptop or a
Raspberry Pi, with a CLI, a web control panel, and an ArduPilot integration.

## What it does

- **Tracking:** one MediaPipe pose network finds faces (nose/eyes/ears) and
  bodies (torso). **Red** cross on each person, **green** on the closest to
  center, **blue** cross at center, with pixel-distance labels.
- **Accurate & light:** One-Euro filtering, visibility-weighted centers,
  threaded latest-frame capture, lite/full models — tuned to run on a Pi.
- **Click-to-follow + detection-assisted:** click any object (a car, etc.) to
  follow it with an OpenCV tracker. With `--detector`, a custom
  **VisDrone-trained YOLOv8** model (run via ONNX Runtime, no PyTorch) snaps the
  click onto the real detected box, **re-locks to fresh detections** so it can't
  silently drift onto the background, and **reports the target LOST** when the
  object is actually gone.
- **Web panel:** multi-camera (webcam + RTSP), live video, an **Auto-best**
  view that switches to whichever camera has the clearest subject, and a
  **Grid** multi-view.
- **UAV:** aim a servo camera **gimbal** at the target, or **orbit-follow** the
  person with ArduPilot — behind conservative safety gates.
- **Custom firmware:** a scaffolded custom ArduPilot Plane build enabling the
  gimbal, camera, follow, and scripting features this project uses.

## Repo map

| Path | What |
|------|------|
| [`tracker.py`](tracker.py) | Desktop/CLI tracker |
| [`app.py`](app.py) | Web control panel (multi-camera, RTSP, auto-best) |
| [`detector.py`](detector.py) | YOLOv8 object detector via ONNX Runtime (no PyTorch) |
| [`uav.py`](uav.py) | MAVLink gimbal aim + orbit-follow |
| [`geo.py`](geo.py) | Camera line-of-sight → ground position math |
| [`test_uav.py`](test_uav.py), [`test_flight.py`](test_flight.py), [`test_detector.py`](test_detector.py) | Tests (no hardware) |
| [`models/`](models) | MediaPipe models + VisDrone YOLOv8 (`visdrone_n.onnx`) |
| [`colab/`](colab) | Colab notebook to train the bigger YOLOv8m on a free GPU |
| [`ardupilot_build/`](ardupilot_build) | Custom ArduPilot firmware config |
| [`docs/`](docs) | All project documentation |

## Docs

- **[docs/PARTS.md](docs/PARTS.md)** — recommended hardware to build the aircraft (nothing bought yet).
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — how the components fit and the tracking pipeline.
- **[docs/USAGE.md](docs/USAGE.md)** — install and run everything (CLI, web, tests, SITL, build).
- **[docs/SAFETY.md](docs/SAFETY.md)** — the UAV safety model — read before flying.

## Quick start

```bash
pip install opencv-python==4.12.0.88 mediapipe pymavlink flask
python3 tracker.py          # desktop tracker
python3 app.py              # web panel at http://127.0.0.1:5000

# detection-assisted follow (drift-proof) with the bundled VisDrone model:
pip install onnxruntime
python3 tracker.py --detector --detect-classes car,van,truck,bus
python3 app.py --detector   # same, in the web panel
```

See [docs/USAGE.md](docs/USAGE.md) for camera, UAV, and web options.

## Custom detector

The bundled [`models/visdrone_n.onnx`](models) is a YOLOv8n fine-tuned on
VisDrone (aerial people + vehicles). To train the larger, more accurate
**yolov8m** on a free Colab GPU (~1–2 h), open
[`colab/train_yolov8m_visdrone.ipynb`](colab/train_yolov8m_visdrone.ipynb) — it
exports CoreML (Mac Neural Engine) and ONNX, which drop into `--detector` the
same way.

> This is a follow/filming platform: it orbits a subject at a standoff distance
> and never flies at them. Keep a pilot in command and test in simulation first
> — see [docs/SAFETY.md](docs/SAFETY.md).
