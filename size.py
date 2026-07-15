"""Real-world size estimation for detected objects.

Two honest modes, best-available:
  - MEASURED: with a metres-per-pixel scale, real size = pixel size x scale.
    The scale comes either from UAV altitude, or -- in a top-down view with no
    telemetry -- is inferred from known-size vehicles acting as rulers (cars are
    ~4.5 m long, etc.). This is independent of any per-object width assumption,
    so it does NOT circularly depend on the monocular distance estimate.
  - PRIOR: with no scale, fall back to the typical footprint for the class.

Monocular + flat-ground assumptions, so treat results as ~+/-25% estimates.
"""

# typical (length, width) footprint in metres, per VisDrone class
CLASS_SIZE_M = {
    "pedestrian": (0.6, 0.5), "people": (0.6, 0.5), "bicycle": (1.7, 0.6),
    "car": (4.5, 1.8), "van": (5.5, 2.0), "truck": (8.0, 2.5),
    "tricycle": (2.6, 1.2), "awning-tricycle": (3.2, 1.4),
    "bus": (12.0, 2.55), "motor": (2.0, 0.8),
}
# reliable rulers: class -> typical LENGTH (m) used to infer the scene scale
RULER_LEN_M = {"car": 4.5, "van": 5.5, "bus": 12.0, "truck": 8.0}
M_TO_FT = 3.280839895


def estimate_scene_scale(dets, min_px=6):
    """Metres-per-pixel for a top-down scene, from the median of known-size
    vehicles used as rulers. `dets` have .name/.w/.h. None if no usable ruler.
    Median makes it robust to a few odd boxes."""
    s = []
    for d in dets:
        length_m = RULER_LEN_M.get(getattr(d, "name", None))
        if length_m is None:
            continue
        pix = max(d.w, d.h)
        if pix > min_px:
            s.append(length_m / pix)
    if not s:
        return None
    s.sort()
    return s[len(s) // 2]


def measure_size(box_w, box_h, scale):
    """(long, short) real dimensions in metres from a metres-per-pixel scale."""
    if not scale or scale <= 0:
        return None
    a, b = box_w * scale, box_h * scale
    return (max(a, b), min(a, b))


def prior_size(name):
    """Typical (length, width) in metres for a class name, or None."""
    return CLASS_SIZE_M.get(str(name).lower())


def object_size(name, box_w, box_h, scale):
    """Best available size: measured from `scale` if given, else the class
    prior. Returns (dims, measured) where measured is True for a real
    measurement, False for a prior; (None, False) if neither is available."""
    if scale:
        dims = measure_size(box_w, box_h, scale)
        if dims:
            return dims, True
    p = prior_size(name)
    return (p, False) if p else (None, False)


def fmt_size(dims, imperial=False):
    """Format (length, width) as e.g. '4.5x1.8m' or '14.8x5.9ft'."""
    if not dims:
        return "?"
    length, width = dims
    if imperial:
        return f"{length * M_TO_FT:.1f}x{width * M_TO_FT:.1f}ft"
    return f"{length:.1f}x{width:.1f}m"
