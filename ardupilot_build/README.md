# Custom ArduPilot build — tracker UAV

A custom ArduPilot **Plane** firmware build tailored to this project (camera
gimbal + follow), scaffolded but **not fully compiled**. It reproduces locally
what the ArduPilot CustomBuild site (custom.ardupilot.org) does: pick features,
generate an `extra_hwdef.dat`, then `waf configure` + `waf plane`.

## Files

| File | Role |
|------|------|
| [`extra_hwdef.dat`](extra_hwdef.dat) | The feature selection — enables gimbal/mount, camera, follow, and Lua scripting. Same artifact CustomBuild emits from its checkboxes. |
| [`build.sh`](build.sh) | Clones ArduPilot at a pinned version and runs `waf configure` with the feature file. Stops before the compile unless you pass `--full`. |
| [`scripts/tracker_failsafe.lua`](scripts/tracker_failsafe.lua) | Optional on-vehicle Lua guard (needs the scripting feature this build enables). |

## Why these features

The build turns on exactly what the tracker/gimbal system in this repo uses:

- **`AP_MOUNT_ENABLED` / `AP_CAMERA_MOUNT_ENABLED` / `HAL_MOUNT_SERVO_ENABLED`** —
  the servo gimbal that `uav.py` aims with `MOUNT_CONTROL`.
- **`AP_CAMERA_ENABLED`** — camera shutter/record control over MAVLink.
- **`AP_FOLLOW_ENABLED`** — the follow library, complementing the companion's
  `DO_REPOSITION` orbit commands.
- **`AP_SCRIPTING_ENABLED`** — Lua, so follow/failsafe logic can also run
  on the autopilot itself (survives a companion-link dropout).

## How to build

Nothing here has been compiled — there's no ARM toolchain on this machine and
you asked not to fully make it. To do it yourself:

```bash
cd ardupilot_build

# 1) configure only (default) — clones ArduPilot, runs waf configure, stops
./build.sh --board CubeOrange

# 2) when ready, actually compile (needs the ARM toolchain)
./build.sh --board CubeOrange --full
#   firmware lands in ardupilot/build/CubeOrange/bin/
```

Other boards: `./build.sh --board MatekF405-Wing`. Software-in-the-loop (no
ARM toolchain, builds with host gcc): `./build.sh --sitl`.

### Toolchain (for a real board build)

Install ArduPilot's prerequisites once:

```bash
# inside the cloned ardupilot/ dir
Tools/environment_install/install-prereqs-mac.sh      # macOS
Tools/environment_install/install-prereqs-ubuntu.sh   # Linux / Pi
```

## Notes

- **Flash:** scripting + gimbal need spare flash. CubeOrange is fine; small F4
  boards may overflow — uncomment the trim section in `extra_hwdef.dat`.
- **Pinned version:** `build.sh` pins `Plane-4.5`; change with `--ref`.
- The `ardupilot/` clone is created next to this README and is intentionally
  git-ignored / not committed (it's large).
