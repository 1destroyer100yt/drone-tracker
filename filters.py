"""Signal smoothing shared across the tracker (One-Euro filter).

Kept in its own module so both tracker.py and motion.py can use it without a
circular import.
"""

import math


class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012) for one scalar signal.

    Adapts its smoothing to speed: heavy smoothing when the value is
    nearly still (no jitter), light smoothing when it moves fast (no lag).
    """

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, t, x):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        # smoothed derivative (px/s)
        a_d = self._alpha(dt, self.d_cutoff)
        dx = (x - self.x_prev) / dt
        dx_s = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_s

        # cutoff rises with speed -> less smoothing when moving fast
        cutoff = self.min_cutoff + self.beta * abs(dx_s)
        a = self._alpha(dt, cutoff)
        self.x_prev = a * x + (1 - a) * self.x_prev
        return self.x_prev


class PointSmoother:
    """One-Euro-filtered (x, y) points, keyed by track name."""

    def __init__(self):
        self.filters = {}

    def update(self, key, t, x, y):
        if key not in self.filters:
            self.filters[key] = (OneEuroFilter(), OneEuroFilter())
        fx, fy = self.filters[key]
        return fx(t, x), fy(t, y)

    def forget_missing(self, live_keys):
        for key in list(self.filters):
            if key not in live_keys:
                del self.filters[key]
