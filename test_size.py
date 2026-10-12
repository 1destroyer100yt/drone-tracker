"""Tests for real-world size estimation (size.py).

  python3 test_size.py
"""

from size import (estimate_scene_scale, measure_size, prior_size,
                  object_size, fmt_size)


class D:
    """Minimal stand-in for a Detection (name/w/h)."""
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h


def test_scene_scale():
    # cars ~4.5 m at ~90 px long -> ~0.05 m/px; median is robust to the ped
    dets = [D("car", 90, 45), D("car", 100, 50), D("van", 110, 55),
            D("pedestrian", 10, 20)]
    s = estimate_scene_scale(dets)
    assert s is not None and 0.04 < s < 0.06, s
    assert estimate_scene_scale([D("pedestrian", 10, 20)]) is None  # no ruler
    assert estimate_scene_scale([]) is None
    print(f"scene scale: OK  ({s:.4f} m/px from vehicle rulers)")


def test_measure_and_prior():
    assert measure_size(90, 45, 0.05) == (4.5, 2.25)
    assert measure_size(90, 45, None) is None
    assert prior_size("car") == (4.5, 1.8) and prior_size("nope") is None
    dims, meas = object_size("car", 90, 45, 0.05)
    assert meas and dims == (4.5, 2.25)
    dims, meas = object_size("car", 90, 45, None)     # no scale -> prior
    assert not meas and dims == (4.5, 1.8)
    dims, meas = object_size("unknownthing", 90, 45, None)
    assert dims is None and not meas
    print("measure / prior: OK  (measured beats prior; prior fallback works)")


def test_fmt():
    assert fmt_size((4.5, 1.8)) == "4.5x1.8m"
    assert fmt_size((4.5, 1.8), imperial=True).endswith("ft")
    assert fmt_size(None) == "?"
    print("fmt: OK")


def main():
    test_scene_scale()
    test_measure_and_prior()
    test_fmt()
    print("\nall size tests passed.")


if __name__ == "__main__":
    main()
