#!/usr/bin/env bash
#
# Custom ArduPilot Plane build for the tracker UAV.
#
# Mirrors what custom.ardupilot.org does under the hood:
#   1. get the ArduPilot source at a pinned version
#   2. `waf configure` for the chosen board WITH our feature extra_hwdef
#   3. `waf plane` to compile   <-- the long step; SKIPPED by default
#
# "Don't fully make it": by default this stops after configure. Pass --full to
# actually run the ~15-30 min compile once you have the toolchain set up.
#
# Usage:
#   ./build.sh                      # clone + configure for CubeOrange, stop
#   ./build.sh --board MatekF405-Wing
#   ./build.sh --sitl               # configure the software-in-the-loop build
#   ./build.sh --full               # also compile (needs the ARM toolchain)
#   ./build.sh --ref Plane-4.5      # pin a different ArduPilot version

set -euo pipefail

BOARD="CubeOrange"
REF="Plane-4.5"          # pinned stable Plane branch/tag
DO_BUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --sitl)  BOARD="sitl"; shift ;;
    --ref)   REF="$2"; shift 2 ;;
    --full)  DO_BUILD=1; shift ;;
    -h|--help) awk 'NR>1 && /^#/{sub(/^# ?/,"");print;next} NR>1{exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
HWDEF="$HERE/extra_hwdef.dat"
AP_DIR="$HERE/ardupilot"

echo "==> board=$BOARD  ref=$REF  full-build=$DO_BUILD"
echo "==> feature file: $HWDEF"

# ---- toolchain sanity (warn, don't fail) ----------------------------------
if [ "$BOARD" != "sitl" ] && ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "!! arm-none-eabi-gcc not found. Configure will still run, but a real"
  echo "!! board build (--full) needs it. Install with ArduPilot's setup:"
  echo "!!   Tools/environment_install/install-prereqs-mac.sh   (or -ubuntu.sh)"
fi

# ---- get the source at the pinned version ---------------------------------
if [ ! -d "$AP_DIR" ]; then
  echo "==> cloning ArduPilot ($REF) into $AP_DIR (shallow, with submodules)"
  git clone --recurse-submodules --shallow-submodules --depth 1 \
    --branch "$REF" https://github.com/ArduPilot/ardupilot "$AP_DIR"
else
  echo "==> reusing existing clone at $AP_DIR"
fi

cd "$AP_DIR"

# ---- configure with our custom feature set --------------------------------
echo "==> waf configure --board $BOARD --extra-hwdef=$HWDEF"
./waf configure --board "$BOARD" --extra-hwdef="$HWDEF"

# ---- compile (only with --full) -------------------------------------------
if [ "$DO_BUILD" = 1 ]; then
  echo "==> waf plane  (compiling; this takes a while)"
  ./waf plane
  echo "==> firmware in: $AP_DIR/build/$BOARD/bin/"
else
  echo
  echo "==> configure complete. Compile was SKIPPED (\"don't fully make it\")."
  echo "==> to finish the build later, run:"
  echo "      cd \"$AP_DIR\" && ./waf plane"
  echo "==> or re-run this script with --full"
fi
