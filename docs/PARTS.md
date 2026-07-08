# Parts guide — person-following camera plane

A hardware shopping list for the tracker in this repo: a fixed-wing aircraft
that carries a Raspberry Pi running the MediaPipe tracker, aims a camera gimbal
at a person, and (optionally) orbits them under ArduPilot. Nothing here is
bought yet — this is a from-scratch recommendation.

Two builds are given: a **recommended** build (reliable, good margins) and a
**budget** build. Prices are rough July-2026 USD for planning only.

> Safety first: this is a follow/filming platform. It orbits a subject at a
> standoff distance and never flies at them (see [SAFETY.md](SAFETY.md)). Fly
> only where local rules allow, keep a human pilot on the sticks, and test in
> simulation before real flight.

---

## The short answer (recommended build)

| Part | Pick | ~USD | Why |
|------|------|------|-----|
| Airframe | **Skywalker 1900 / Believer 1960 twin** | 120–260 | Big payload bay + long endurance for a Pi + gimbal |
| Flight controller | **Hex CubeOrange+** | 250 | Lots of flash for the scripting/gimbal custom build |
| Companion computer | **Raspberry Pi 5 (4 GB) + active cooler** | 80 | Enough CPU for smooth MediaPipe tracking |
| Camera | **Pi Camera Module 3 (Wide)** | 35 | Autofocus; set `--hfov` to match the lens |
| Gimbal | **2-axis servo pan/tilt + 2× digital MG-class servos** | 40 | Matches the servo-mount feature in the build |
| GPS/compass | **Holybro M10** | 45 | Modern GNSS, needed for follow/orbit |
| Telemetry | **Holybro SiK 915 MHz (US) / 433 MHz (EU)** | 40 | GCS link + MAVLink to the Pi |
| RC link | **RadioMaster TX16S + ELRS RX** | 200 | Long-range, reliable manual override |
| Power module | **Holybro PM02 / PM07** | 30 | FC power + battery current sensing |
| Pi power | **5 V 5 A UBEC (dedicated)** | 12 | Pi 5 needs a clean, separate 5 V rail |
| Battery | **4S 8000–10000 mAh LiPo** | 70 | Endurance for the extra payload |
| Misc | props, XT60/90, foam mounts, microSD, wiring | 40 | — |

**Ballpark total (excluding radio you may own): ~$700–850.**

---

## Category detail

### Airframe (fixed-wing)
A camera platform needs payload room and stable, slow flight — not a hot 3D
flyer.

- **Recommended:** *Skywalker 1900* (single motor, huge nose bay) or
  *Believer 1960* (twin motor, more payload + redundancy, long endurance).
  Both are proven ArduPlane camera/mapping platforms.
- **Budget / first build:** *Volantex Ranger 2000* — cheap, forgiving, big
  fuselage for the Pi and battery.
- **Avoid** small flying wings and racers: no room for a Pi + gimbal and too
  twitchy for filming.

### Flight controller
- **Recommended:** *Hex CubeOrange+*. It has the flash headroom the
  [custom build](../ardupilot_build/README.md) needs for gimbal + Lua
  scripting, plus triple-redundant IMUs. This is what the build config targets.
- **Value:** *Matek H743-WING V3* — H7 chip, wing-oriented layout, enough flash
  for scripting at a much lower price. If you use it, build with
  `./build.sh --board MatekH743`.
- Small F4 boards work but may overflow flash with scripting on — see the trim
  section in [`extra_hwdef.dat`](../ardupilot_build/extra_hwdef.dat).

### Companion computer (runs the tracker)
MediaPipe pose is CPU-heavy, so this drives your tracking FPS.

- **Recommended:** *Raspberry Pi 5 (4 GB)* with an **active cooler** — best
  real-time FPS with the full model.
- **Budget:** *Raspberry Pi 4 (4 GB)* — works well with `--model lite`.
- **Too weak:** Pi Zero 2 W. Don't.
- Must run 64-bit Raspberry Pi OS (MediaPipe requirement). See
  [USAGE.md](USAGE.md).

### Camera
- **Pi Camera Module 3** (standard ~66° or **Wide** ~102°). Autofocus keeps a
  moving subject sharp.
- **USB alternative:** *Logitech C920* — plug-and-play, ~70° FOV.
- **IP/RTSP alternative:** any RTSP camera — the app accepts an RTSP URL as a
  source directly.
- **Whatever you pick, set `--hfov` to its horizontal field of view** so the
  gimbal angles and distance math are correct.

### Gimbal (aims the camera)
The custom build enables the **servo mount** backend, so the simplest match is:

- **Recommended:** a 2-axis (pan/tilt) bracket driven by **2 digital metal-gear
  servos** (MG996R-class or better). Wire pan/tilt to FC servo outputs and set
  the `MNT1_TYPE`/`SERVOn_FUNCTION` mount params.
- **Upgrade:** a brushless *STorM32* gimbal for smooth stabilized footage — note
  that's a serial-mount gimbal, so you'd enable the serial mount backend instead
  of servo.

### GPS / compass
- **Holybro M10** (or M9N). Required for follow/orbit — the geo-projection in
  [`geo.py`](../geo.py) turns the camera angle into a lat/lon using the
  aircraft's GPS.

### Telemetry & RC
- **Telemetry:** *Holybro SiK 915 MHz* (US) / *433 MHz* (EU) for the ground
  station and to bridge MAVLink. On the aircraft the Pi connects to the FC over
  **serial (TELEM2)**, not the radio.
- **RC:** *RadioMaster TX16S* + an **ExpressLRS** receiver — long range and a
  rock-solid manual override, which the safety model depends on.

### Power
- **FC:** *Holybro PM02* (or PM07 if using their power distribution) for clean
  FC power and current sensing.
- **Pi:** a **dedicated 5 V 5 A UBEC** — the Pi 5 can draw 3 A+, so give it its
  own regulator, not the FC's BEC.
- **Battery:** *4S 8000–10000 mAh* LiPo for the recommended airframes; size to
  your endurance target.

---

## How it wires together

```
[Pi Camera / USB / RTSP] --> Raspberry Pi (tracker.py / app.py)
                                   |  USB Wi-Fi = web UI on your phone/laptop
                                   |  serial UART (GPIO14/15) @ 921600
                                   v
                          Flight Controller (ArduPlane, TELEM2)
                             |            |            |
                          Gimbal       GPS/M10      SiK telem --> ground station
                          servos                    ELRS RX  <-- TX16S (pilot)
```

- **Pi ↔ FC serial:** Pi TX/RX (GPIO 14/15, 3.3 V) to the FC's TELEM2 RX/TX
  (also 3.3 V — safe, no level shifter). Set that FC serial port to MAVLink2 at
  921600. Then run `--mavlink /dev/serial0 --baud 921600`.
- **Gimbal:** pan/tilt servos to FC servo rails; assign them as mount outputs in
  ArduPilot params.
- Give the Pi its **own 5 V UBEC**; share only ground with the FC on the serial
  link.

---

## How parts map to the software / build

| Choice | Affects |
|--------|---------|
| Camera FOV | `--hfov` value (default 62.2° = Pi Cam v2) |
| Pi model (4 vs 5) | use `--model lite` vs `full`; expected FPS |
| FC flash size | whether scripting fits — see build trim section |
| Servo gimbal | `HAL_MOUNT_SERVO_ENABLED` in [`extra_hwdef.dat`](../ardupilot_build/extra_hwdef.dat) |
| GPS present | required for `--follow` orbit (geo-projection) |
| Pi↔FC baud | `--baud` (recommend 921600 on serial) |

---

## Cost summary

| Build | Rough total (no RC radio) |
|-------|---------------------------|
| Recommended | ~$700–850 |
| Budget (Pi 4 + Matek H743 + Ranger 2000 + USB cam) | ~$450–550 |

See [USAGE.md](USAGE.md) to bring the software up on the Pi and
[SAFETY.md](SAFETY.md) before any flight.
