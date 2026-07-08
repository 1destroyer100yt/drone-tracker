"""Benchmark the click-to-follow trackers (CSRT, KCF, DROTRACK) on a synthetic
moving-object sequence, reporting speed (fps) and tracking error (px).

No camera or network needed. It renders a textured square that translates and
slowly scales (mimicking a car seen from a moving drone), tracks it with each
backend, and compares the tracked centre to ground truth.

Run:
  python3 benchmark_trackers.py
"""

import math
import time

import cv2
import numpy as np

import tracker as T

W, H, N = 640, 480, 120


def render(i):
    """Frame i: a textured square moving right + drifting down, slowly growing.
    Returns (frame, (cx, cy)) ground-truth center."""
    f = np.full((H, W, 3), 30, np.uint8)
    cx = 90 + i * 3.2
    cy = 240 + 40 * math.sin(i / 25.0)
    s = int(46 + i * 0.15)
    x0, y0 = int(cx - s / 2), int(cy - s / 2)
    cv2.rectangle(f, (x0, y0), (x0 + s, y0 + s), (170, 170, 175), -1)
    # texture so optical-flow / correlation trackers have features
    for dx in range(-s // 2 + 6, s // 2 - 5, 9):
        for dy in range(-s // 2 + 6, s // 2 - 5, 9):
            cv2.circle(f, (int(cx + dx), int(cy + dy)), 2, (70, 70, 70), -1)
    return f, (cx, cy)


def bench(algo):
    fol = T.ObjectFollower(box_size=54, algo=algo)
    f0, (cx, cy) = render(0)
    if fol.start(f0, cx, cy) is False:
        return None
    errs, frames, t0 = [], 0, time.perf_counter()
    for i in range(1, N):
        f, (gx, gy) = render(i)
        c = fol.update(f)
        frames += 1
        if c is not None:
            errs.append(math.hypot(c[0] - gx, c[1] - gy))
    dt = time.perf_counter() - t0
    if not errs:
        return dict(algo=algo, fps=frames / dt, lost=True)
    return dict(algo=algo, fps=frames / dt, mean_err=sum(errs) / len(errs),
                max_err=max(errs), tracked=len(errs), total=N - 1)


def main():
    print(f"sequence: {N} frames, {W}x{H}, moving+scaling textured object\n")
    print(f"{'tracker':10s} {'fps':>7s} {'mean_err':>9s} {'max_err':>8s} "
          f"{'tracked':>8s}")
    for algo in ("CSRT", "KCF", "DROTRACK"):
        r = bench(algo)
        if r is None:
            print(f"{algo:10s}   failed to initialize")
        elif r.get("lost"):
            print(f"{algo:10s} {r['fps']:7.0f}   (lost the target)")
        else:
            print(f"{algo:10s} {r['fps']:7.0f} {r['mean_err']:8.1f}px "
                  f"{r['max_err']:7.1f}px {r['tracked']:4d}/{r['total']}")
    print("\nlower err = more accurate; higher fps = faster. Pick per your CPU.")


if __name__ == "__main__":
    main()
