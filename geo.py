"""Geometry for turning a camera line-of-sight into a ground target position.

Pure functions, no MAVLink, so the flight math can be unit-tested on its own.

Model & assumptions (documented on purpose -- validate in SITL before flying):
  - Flat ground a known height (AGL) below the aircraft.
  - The target sits on that ground plane.
  - Angles are radians. Headings are compass-style: 0 = North, +East.
  - Depression is the angle of the line-of-sight BELOW horizontal (positive
    down). A near-horizontal LOS never reliably hits the ground, so we refuse
    to project it rather than return a wild far-away point.
"""

import math

EARTH_M_PER_DEG_LAT = 111320.0     # metres per degree of latitude (approx)
MIN_DEPRESSION_RAD = math.radians(3.0)  # below this the ground hit is unreliable


def line_of_sight(aircraft_yaw, aircraft_pitch, gimbal_yaw, gimbal_pitch):
    """Compose the aircraft attitude with the gimbal aim into an absolute
    line-of-sight (heading, depression), all radians.

    aircraft_yaw     : compass heading of the nose (0=N, +E)
    aircraft_pitch   : nose-up positive
    gimbal_yaw       : camera yaw right of the nose, positive right
    gimbal_pitch     : camera tilt below the nose axis, positive down
    """
    heading = (aircraft_yaw + gimbal_yaw) % (2 * math.pi)
    depression = gimbal_pitch - aircraft_pitch  # nose-up reduces depression
    return heading, depression


def project_target(lat, lon, alt_agl, heading, depression):
    """Ground intersection of the LOS. Returns (lat, lon, ground_range_m) or
    None if the LOS is too shallow / above horizontal to hit the ground."""
    if depression < MIN_DEPRESSION_RAD:
        return None
    ground_range = alt_agl / math.tan(depression)
    north = ground_range * math.cos(heading)
    east = ground_range * math.sin(heading)
    dlat = north / EARTH_M_PER_DEG_LAT
    dlon = east / (EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon, ground_range


def project_from_range(lat, lon, heading, depression, slant_range):
    """Ground position of a target at a KNOWN slant range (metres) along the
    line-of-sight. Unlike project_target this needs no flat-ground assumption --
    the horizontal offset is range*cos(depression). Returns (lat, lon, horiz_m).

    Use this when a monocular distance estimate (distance.py) gives the range to
    the person; it is more robust than the flat-ground projection over uneven
    terrain."""
    horiz = slant_range * math.cos(depression)
    north = horiz * math.cos(heading)
    east = horiz * math.sin(heading)
    dlat = north / EARTH_M_PER_DEG_LAT
    dlon = east / (EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon, horiz


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lat/lon points."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))
