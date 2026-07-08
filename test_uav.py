"""Test the UAV gimbal link without any hardware, plane, or camera.

Spins up a local UDP MAVLink endpoint, feeds MavlinkUAV a simulated target
moving in a circle around the frame, and decodes the MOUNT_CONTROL messages
that come back out -- checking that the gimbal pitch/yaw angles match the
target's pixel offset. Confirms the camera-only aim path end to end.

Run:
  python3 test_uav.py            # assertions + a live readout
"""

import math
import time

from pymavlink import mavutil

from uav import MavlinkUAV

W, H = 640, 480
CENTER = (W / 2.0, H / 2.0)
HFOV = 62.2  # deg, Pi Camera v2


def expected_angles(target, center, w, h, hfov_deg):
    """Reference math, independent of uav.py, for cross-checking."""
    hfov = math.radians(hfov_deg)
    vfov = hfov * (h / w)
    ax = (target[0] - center[0]) / (w / 2.0) * (hfov / 2.0)
    ay = (target[1] - center[1]) / (h / 2.0) * (vfov / 2.0)
    # MOUNT_CONTROL is sent as pitch=-ay, yaw=ax (centi-degrees)
    return math.degrees(-ay), math.degrees(ax)  # (pitch_deg, yaw_deg)


def main():
    recv = mavutil.mavlink_connection("udpin:127.0.0.1:14552")
    uav = MavlinkUAV("udpout:127.0.0.1:14552", hfov_deg=HFOV)

    # --- static corner checks ---
    cases = {
        "center":      (W / 2, H / 2),
        "right edge":  (W, H / 2),
        "left edge":   (0, H / 2),
        "top edge":    (W / 2, 0),
        "bottom edge": (W / 2, H),
    }
    print("static target checks (pitch, yaw in degrees):")
    for name, target in cases.items():
        uav.send_gimbal(target, CENTER, (W, H))
        msg = recv.recv_match(type="MOUNT_CONTROL", blocking=True, timeout=3)
        assert msg is not None, f"no MOUNT_CONTROL for '{name}'"
        got_pitch = msg.input_a / 100.0   # centi-deg -> deg
        got_yaw = msg.input_c / 100.0
        exp_pitch, exp_yaw = expected_angles(target, CENTER, W, H, HFOV)
        assert abs(got_pitch - exp_pitch) < 0.05, (name, got_pitch, exp_pitch)
        assert abs(got_yaw - exp_yaw) < 0.05, (name, got_yaw, exp_yaw)
        print(f"  {name:12s} pitch={got_pitch:+7.2f}  yaw={got_yaw:+7.2f}")

    # sanity on signs: target to the right -> +yaw, target down -> -pitch
    assert expected_angles((W, H / 2), CENTER, W, H, HFOV)[1] > 0
    assert expected_angles((W / 2, H), CENTER, W, H, HFOV)[0] < 0

    # --- simulated moving target (circle), like a person orbiting center ---
    print("\nsimulated moving target (10 frames of a circular sweep):")
    radius = 150
    for i in range(10):
        ang = 2 * math.pi * i / 10
        target = (CENTER[0] + radius * math.cos(ang),
                  CENTER[1] + radius * math.sin(ang))
        uav.send_gimbal(target, CENTER, (W, H))
        msg = recv.recv_match(type="MOUNT_CONTROL", blocking=True, timeout=3)
        assert msg is not None
        exp_pitch, exp_yaw = expected_angles(target, CENTER, W, H, HFOV)
        assert abs(msg.input_a / 100.0 - exp_pitch) < 0.05
        assert abs(msg.input_c / 100.0 - exp_yaw) < 0.05
        print(f"  frame {i}: target=({target[0]:5.0f},{target[1]:5.0f})  "
              f"pitch={msg.input_a/100.0:+7.2f}  yaw={msg.input_c/100.0:+7.2f}")
        time.sleep(0.02)

    uav.close()
    recv.close()
    print("\nALL TESTS PASSED - gimbal aims correctly, aircraft never commanded")


if __name__ == "__main__":
    main()
