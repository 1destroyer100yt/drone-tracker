"""Tests for the monocular person-distance estimator (distance.py).

Verifies the pinhole model end to end with synthetic pose landmarks: a subject
of known real size placed at a known distance must measure back to that
distance.
"""

import math

import distance


class LM:
    """Minimal stand-in for a MediaPipe normalized landmark."""
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def make_pose(shoulder_px=None, eye_px=None, w=640, h=480,
              shoulder_vis=1.0, eye_vis=1.0):
    """Build a 33-landmark pose with the shoulders / eyes a given pixel span
    apart, centered, everything else invisible."""
    lms = [LM(0.5, 0.5, 0.0) for _ in range(33)]
    if shoulder_px is not None:
        dx = (shoulder_px / w) / 2
        lms[11] = LM(0.5 - dx, 0.5, shoulder_vis)   # left shoulder
        lms[12] = LM(0.5 + dx, 0.5, shoulder_vis)   # right shoulder
    if eye_px is not None:
        dx = (eye_px / w) / 2
        lms[2] = LM(0.5 - dx, 0.5, eye_vis)         # left eye
        lms[5] = LM(0.5 + dx, 0.5, eye_vis)         # right eye
    return lms


def test_focal():
    # 640 px wide, 62.2 deg HFOV -> f = 320 / tan(31.1deg)
    f = distance.focal_px(640, math.radians(62.2))
    assert abs(f - 320.0 / math.tan(math.radians(31.1))) < 1e-6
    print(f"focal_px OK: {f:.1f}px for 640w / 62.2deg")


def test_shoulder_distance():
    w, h, hfov = 640, 480, math.radians(62.2)
    f = distance.focal_px(w, hfov)
    # place a 0.40 m subject at 3.0 m -> expected pixel width = 0.40*f/3.0
    px = 0.40 * f / 3.0
    pose = make_pose(shoulder_px=px, w=w, h=h)
    d, method = distance.estimate_distance(pose, w, h, hfov)
    assert method == "shoulders"
    assert abs(d - 3.0) < 1e-6, d
    print(f"shoulder distance OK: measured {d:.3f} m at true 3.0 m")

    # doubling the pixel width halves the distance
    pose2 = make_pose(shoulder_px=2 * px, w=w, h=h)
    d2, _ = distance.estimate_distance(pose2, w, h, hfov)
    assert abs(d2 - 1.5) < 1e-6, d2
    print(f"inverse-law OK: 2x pixels -> {d2:.3f} m (half)")


def test_eye_fallback():
    w, h, hfov = 640, 480, math.radians(62.2)
    f = distance.focal_px(w, hfov)
    # shoulders not visible -> falls back to eyes (0.063 m) at 1.5 m
    px = 0.063 * f / 1.5
    pose = make_pose(eye_px=px, w=w, h=h, shoulder_vis=0.0)
    d, method = distance.estimate_distance(pose, w, h, hfov)
    assert method == "eyes"
    assert abs(d - 1.5) < 1e-6, d
    print(f"eye fallback OK: measured {d:.3f} m at true 1.5 m")


def test_fmt():
    assert distance.fmt_distance(3.2) == "3.2m"
    assert distance.fmt_distance(None) is None
    # 3.2 m = 125.98 in = 10 ft 6 in
    assert distance.fmt_distance(3.2, imperial=True) == "10ft 6in"
    # 0.3048 m = exactly 1 ft 0 in
    assert distance.fmt_distance(0.3048, imperial=True) == "1ft 0in"
    # rounding that would hit 12 in rolls to the next foot
    assert distance.fmt_distance(0.999 * 12 * 0.0254, imperial=True) == "1ft 0in"
    print("fmt_distance OK: metric + feet/inches + rounding rollover")


def test_nothing_visible():
    w, h, hfov = 640, 480, math.radians(62.2)
    pose = make_pose(shoulder_vis=0.0, eye_vis=0.0)  # nothing usable
    assert distance.estimate_distance(pose, w, h, hfov) is None
    print("no visible segment -> None OK")


if __name__ == "__main__":
    test_focal()
    test_shoulder_distance()
    test_eye_fallback()
    test_fmt()
    test_nothing_visible()
    print("\nALL DISTANCE TESTS PASSED")
