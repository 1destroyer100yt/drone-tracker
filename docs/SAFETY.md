# Safety model

This platform can command a fixed-wing aircraft. The follow feature is designed
to be conservative: it circles a subject at a distance and never flies at a
person. Read this before enabling `--follow` or flying.

## What each mode can and cannot do

| Mode | Sends | Moves the aircraft? |
|------|-------|---------------------|
| Tracking only (no `--mavlink`) | nothing | No |
| Gimbal aim (`--mavlink`) | `MOUNT_CONTROL` | No — camera only |
| Orbit-follow (`--follow`) | `DO_REPOSITION` | Yes — loiters around the target |

## Follow safety gates (all enforced in `uav.py`, tested in `test_flight.py`)

The orbit command is sent **only** when every condition holds:

1. **Armed** — refuses if disarmed.
2. **GUIDED mode** — refuses in any other mode. It never changes mode and never
   arms the vehicle.
3. **Above `--min-alt`** — refuses below the altitude floor (default 30 m AGL).
4. **Fresh telemetry** — refuses if position/attitude data is stale (>2 s).
5. **Inside `--geofence`** — refuses if the estimated target is farther than the
   fence radius from home (default 300 m).
6. **Usable geometry** — refuses if the camera line-of-sight is too shallow to
   locate the ground reliably.

It also **never** overrides RC, so the pilot can take the sticks at any moment.
On **lost target** it stops commanding and the plane holds its current guided
orbit; airframe recovery (RTL etc.) is left to the pilot and ArduPilot's own
failsafes.

The aircraft orbits at `--orbit-radius` (default 80 m) — a **standoff distance**.
It is not a system that approaches or homes on a person.

## Assumptions & limitations

- **Flat ground** and a **body-fixed camera** are assumed by the geo-projection
  in `geo.py`. Sloped terrain or a stabilized gimbal will bias the target
  estimate. Set `--cam-tilt` to your actual fixed camera down-angle.
- Target position accuracy depends on good **altitude and attitude telemetry**.
- Multi-person tracking (`--num-poses` > 1) does not guarantee stable identity
  between frames; for single-target follow use `--num-poses 1` (the default).

## Pre-flight checklist

1. Test the exact follow settings in **ArduPilot SITL** first (see
   [USAGE.md](USAGE.md)).
2. Confirm a working **manual override** (RC link) and a mode switch to exit
   GUIDED.
3. Set a real **geofence** and **failsafes** in ArduPilot itself — the script's
   gates are a second layer, not a replacement.
4. Verify `--hfov` and `--cam-tilt` match the installed camera.
5. Fly only where local regulations permit; keep the aircraft in sight and a
   pilot in command.

## Scope

This is a follow/filming and search-style platform (aerial cinematography,
following an athlete, inspection). Use it accordingly and within the law.
