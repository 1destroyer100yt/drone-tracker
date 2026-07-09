"""Object velocity / speed estimation for the follower.

Turns the per-frame target centre into a smoothed velocity (pixels/second), and
converts to real-world speed (m/s, mph) when a metres-per-pixel scale is known.
The velocity also feeds occlusion coasting / re-identification (see reid.py).
"""

import math

from filters import OneEuroFilter

MPS_TO_MPH = 2.2369362920544


class VelocityTracker:
    """Smoothed velocity of a moving point. Feed (t, x, y) each frame and read
    velocity in px/s. One-Euro smoothing keeps it responsive without jitter, and
    it works off real timestamps so it's correct on variable-rate video."""

    def __init__(self, min_cutoff=0.6, beta=0.03):
        self._fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.t_prev = None
        self.p_prev = None
        self.vx = 0.0
        self.vy = 0.0

    def update(self, t, x, y):
        if self.t_prev is None:
            self.t_prev, self.p_prev = t, (x, y)
            return 0.0
        dt = max(t - self.t_prev, 1e-6)
        self.vx = self._fx(t, (x - self.p_prev[0]) / dt)
        self.vy = self._fy(t, (y - self.p_prev[1]) / dt)
        self.t_prev, self.p_prev = t, (x, y)
        return self.speed_px

    @property
    def speed_px(self):
        """Smoothed speed magnitude in pixels/second."""
        return math.hypot(self.vx, self.vy)

    def predict(self, dt):
        """Predicted (dx, dy) displacement over dt seconds at current velocity."""
        return self.vx * dt, self.vy * dt


def meters_per_px(distance_m, focal_px):
    """Ground sampling scale at a target `distance_m` away, given the camera's
    focal length in pixels: how many metres one pixel spans at that range."""
    if not distance_m or not focal_px or focal_px <= 0:
        return None
    return distance_m / focal_px


def real_speed(speed_px, distance_m, focal_px):
    """Convert a pixel speed to (m/s, mph) using the metres-per-pixel scale.
    Returns (None, None) when the distance/focal aren't known."""
    mpp = meters_per_px(distance_m, focal_px)
    if mpp is None:
        return None, None
    mps = speed_px * mpp
    return mps, mps * MPS_TO_MPH
