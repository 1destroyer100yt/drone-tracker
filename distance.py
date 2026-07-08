"""Estimate how far a person is from the camera, from a single image.

Monocular distance via the pinhole model: an object of known real width W that
appears P pixels wide sits at

    distance = W * focal_px / P

where focal_px = (image_width / 2) / tan(HFOV / 2) -- the same camera model the
gimbal/geo code already uses, so it needs no extra calibration beyond --hfov.

We measure a body segment whose real size is fairly consistent across adults:
  - shoulder width (landmarks 11,12), ~0.40 m -- big in pixels, low noise, but
    shrinks when the person turns sideways (overestimates distance then)
  - inter-eye distance (landmarks 2,5), ~0.063 m -- very consistent, but small
    in pixels so noisy far away; used as a fallback when shoulders aren't seen

The result is an ESTIMATE (±~15-20%): it depends on the person's real size and
orientation. Good enough for a range readout and to seed the follow target, not
a substitute for a rangefinder / stereo depth.
"""

import math

DEFAULT_SHOULDER_M = 0.40      # average adult bi-acromial width
DEFAULT_EYE_M = 0.063          # average adult inter-pupillary distance

L_SHOULDER, R_SHOULDER = 11, 12
L_EYE, R_EYE = 2, 5


def fmt_distance(m, imperial=False):
    """Format a distance in metres for display. imperial -> feet and inches
    (e.g. '10ft 6in'); metric -> metres (e.g. '3.2m'). None -> None."""
    if m is None:
        return None
    if imperial:
        total_in = m * 39.37007874
        ft = int(total_in // 12)
        inch = int(round(total_in - ft * 12))
        if inch == 12:            # rounding pushed inches to a full foot
            ft += 1
            inch = 0
        return f"{ft}ft {inch}in"
    return f"{m:.1f}m"


def focal_px(image_width, hfov_rad):
    """Focal length in pixels from image width and horizontal FOV."""
    return (image_width / 2.0) / math.tan(hfov_rad / 2.0)


def _seg_px(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def estimate_distance(landmarks, w, h, hfov_rad,
                      shoulder_m=DEFAULT_SHOULDER_M, eye_m=DEFAULT_EYE_M,
                      vis_thresh=0.5):
    """Return (distance_m, method) for one pose, or None if no usable segment.
    Tries shoulders first (robust), falls back to eyes (close range)."""
    f = focal_px(w, hfov_rad)

    ls, rs = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
    if ls.visibility > vis_thresh and rs.visibility > vis_thresh:
        px = _seg_px(ls, rs, w, h)
        if px > 1.0:
            return shoulder_m * f / px, "shoulders"

    le, re = landmarks[L_EYE], landmarks[R_EYE]
    if le.visibility > vis_thresh and re.visibility > vis_thresh:
        px = _seg_px(le, re, w, h)
        if px > 1.0:
            return eye_m * f / px, "eyes"

    return None
