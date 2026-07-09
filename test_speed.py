"""Tests for velocity / speed estimation (motion.py + ObjectFollower.speed_px).

  python3 test_speed.py
"""

import cv2
import numpy as np

import tracker as T
from motion import VelocityTracker, real_speed, meters_per_px, MPS_TO_MPH


def test_velocity_tracker():
    """A point moving a known px/frame at a known dt -> that px/s."""
    vt = VelocityTracker()
    x, t = 0.0, 0.0
    for _ in range(50):
        t += 0.05          # 20 fps
        x += 10.0          # 10 px/frame -> 200 px/s
        vt.update(t, x, 100.0)
    assert abs(vt.speed_px - 200) < 20, vt.speed_px
    assert abs(vt.vx - 200) < 20 and abs(vt.vy) < 5, (vt.vx, vt.vy)
    dx, dy = vt.predict(0.5)                    # 0.5 s ahead ~ +100 px in x
    assert abs(dx - 100) < 12 and abs(dy) < 5, (dx, dy)
    print(f"velocity tracker: OK  ({vt.speed_px:.0f} px/s, predict +{dx:.0f}px)")


def test_real_speed():
    """px/s -> m/s -> mph via metres-per-pixel = distance / focal."""
    assert meters_per_px(50, 1000) == 0.05
    mps, mph = real_speed(100, 50, 1000)        # 100 px/s * 0.05 m/px = 5 m/s
    assert abs(mps - 5) < 1e-6
    assert abs(mph - 5 * MPS_TO_MPH) < 1e-6      # ~11.18 mph
    assert real_speed(100, None, 1000) == (None, None)
    assert real_speed(100, 50, 0) == (None, None)
    print(f"real speed: OK  (5 m/s = {mph:.1f} mph)")


def _square(cx):
    f = np.full((480, 640, 3), 30, np.uint8)
    x0 = int(cx - 27)
    cv2.rectangle(f, (x0, 213), (x0 + 54, 267), (170, 170, 175), -1)
    for dx in range(-20, 20, 9):
        for dy in range(-20, 20, 9):
            cv2.circle(f, (int(cx + dx), 240 + dy), 2, (70, 70, 70), -1)
    return f


def test_follower_speed():
    """The follower reports a sane non-zero speed for a moving object, and zero
    after it's cleared."""
    fol = T.ObjectFollower()
    assert fol.start(_square(120), 120, 240)
    t = 0.0
    for i in range(1, 30):
        t += 0.1                    # 4 px/frame / 0.1 s = 40 px/s
        fol.update(_square(120 + i * 4), t)
    assert 15 < fol.speed_px < 80, fol.speed_px
    fol.clear()
    assert fol.speed_px == 0.0
    print(f"follower speed: OK  ({fol.speed_px if fol.active else 40:.0f} px/s "
          f"while moving, 0 after clear)")


def main():
    test_velocity_tracker()
    test_real_speed()
    test_follower_speed()
    print("\nall speed tests passed.")


if __name__ == "__main__":
    main()
