"""Tests for the advanced-flight (orbit-follow) path -- no vehicle, no network.

Covers two things that matter for safety:
  1. The geo-projection math puts the target where the geometry says it should.
  2. Every safety gate actually blocks an orbit command; a command is only
     emitted when armed + GUIDED + above min alt + inside the geofence.

A fake MAVLink master records commands instead of sending them.
"""

import math

import geo
from uav import MavlinkUAV, FollowConfig

W, H = 640, 480
CENTER = (W / 2.0, H / 2.0)


# ---- 1. geo-projection math -------------------------------------------------
def test_geo():
    lat0, lon0, agl = 47.0, 8.0, 100.0

    # straight down -> target directly below, ~zero ground range
    hd, dep = geo.line_of_sight(0.0, -math.pi / 2, 0.0, 0.0)  # nose pitched down 90
    r = geo.project_target(lat0, lon0, agl, math.radians(0), math.radians(90))
    assert r is not None and r[2] < 1e-6, r
    assert abs(r[0] - lat0) < 1e-9 and abs(r[1] - lon0) < 1e-9

    # 45 deg depression, heading North -> 100 m North
    r = geo.project_target(lat0, lon0, agl, math.radians(0), math.radians(45))
    assert abs(r[2] - 100.0) < 1e-6, r
    assert abs(r[0] - (lat0 + 100.0 / geo.EARTH_M_PER_DEG_LAT)) < 1e-9
    assert abs(r[1] - lon0) < 1e-9

    # 45 deg depression, heading East -> 100 m East, latitude unchanged
    r = geo.project_target(lat0, lon0, agl, math.radians(90), math.radians(45))
    assert abs(r[0] - lat0) < 1e-9
    exp_dlon = 100.0 / (geo.EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat0)))
    assert abs(r[1] - (lon0 + exp_dlon)) < 1e-7, r

    # near-horizontal LOS -> refuse (no reliable ground hit)
    assert geo.project_target(lat0, lon0, agl, 0.0, math.radians(1)) is None
    print("geo projection: down / north / east / horizontal-refuse all OK")


def test_geo_from_range():
    lat0, lon0 = 47.0, 8.0
    # known slant range 100 m, level LOS North -> 100 m North, no flat-ground
    r = geo.project_from_range(lat0, lon0, math.radians(0), math.radians(0), 100)
    assert abs(r[2] - 100.0) < 1e-6, r
    assert abs(r[0] - (lat0 + 100.0 / geo.EARTH_M_PER_DEG_LAT)) < 1e-9
    # 60 deg depression, range 100 -> horizontal 50 m
    r = geo.project_from_range(lat0, lon0, math.radians(0), math.radians(60), 100)
    assert abs(r[2] - 50.0) < 1e-6, r
    # range-based projection works even where flat-ground refuses (shallow LOS)
    r = geo.project_from_range(lat0, lon0, math.radians(90), math.radians(1), 200)
    assert r is not None and abs(r[0] - lat0) < 1e-9
    print("geo-from-range OK: level / 60deg / shallow-LOS all projected")


# ---- fake MAVLink master ----------------------------------------------------
class FakeMav:
    def __init__(self):
        self.orbits = []

    def command_int_send(self, tsys, tcomp, frame, cmd, cur, autoc,
                         p1, p2, p3, p4, x, y, z):
        # DO_REPOSITION: p1 groundspeed, p3 loiter radius
        self.orbits.append(dict(cmd=cmd, speed=p1, radius=p3,
                                lat=x / 1e7, lon=y / 1e7, alt=z))

    def mount_control_send(self, *a):
        pass


class FakeMaster:
    def __init__(self):
        self.mav = FakeMav()
        self.target_system = 1
        self.target_component = 1

    def close(self):
        pass


def make_uav(**cfg):
    import time
    u = MavlinkUAV.__new__(MavlinkUAV)   # bypass real connection
    u.hfov = math.radians(62.2)
    u.vfov = None
    u.master = FakeMaster()
    u.cfg = FollowConfig(**cfg)
    u.armed = True
    u.mode = "GUIDED"
    u.lat, u.lon, u.rel_alt = 47.0, 8.0, 100.0
    u.yaw, u.pitch = 0.0, math.radians(-30)  # nose 30 deg down
    u.home = (47.0, 8.0)
    u.last_telemetry = time.monotonic()
    u._last_cmd_t = 0.0
    u._last_center = None
    return u


# ---- 2. safety gates --------------------------------------------------------
def test_safety_gates():
    # baseline: everything safe, target centred -> orbit IS commanded
    u = make_uav(min_alt=30.0, geofence_radius=100000.0)
    status = u.follow_target(CENTER, CENTER, (W, H))
    assert u.master.mav.orbits, f"expected an orbit command, got: {status}"
    assert status.startswith("orbit"), status
    base_orbit = u.master.mav.orbits[-1]
    assert base_orbit["radius"] == u.cfg.orbit_radius
    print("safe case -> orbit commanded:", status)

    # disarmed -> no command
    u = make_uav(); u.armed = False
    assert u.follow_target(CENTER, CENTER, (W, H)) == "disarmed"
    assert not u.master.mav.orbits

    # wrong mode -> no command
    u = make_uav(); u.mode = "MANUAL"
    assert u.follow_target(CENTER, CENTER, (W, H)).startswith("not GUIDED")
    assert not u.master.mav.orbits

    # below min altitude -> no command
    u = make_uav(min_alt=30.0); u.rel_alt = 10.0
    assert u.follow_target(CENTER, CENTER, (W, H)).startswith("below min alt")
    assert not u.master.mav.orbits

    # stale telemetry -> no command
    u = make_uav(); u.last_telemetry -= 10.0
    assert u.follow_target(CENTER, CENTER, (W, H)) == "no telemetry"
    assert not u.master.mav.orbits

    # target outside geofence -> no command
    u = make_uav(geofence_radius=5.0)   # 5 m fence, target ~170 m away
    assert u.follow_target(CENTER, CENTER, (W, H)) == "target outside geofence"
    assert not u.master.mav.orbits

    print("gates OK: disarmed / wrong-mode / low-alt / stale / geofence all blocked")


def test_rate_limit_and_lost():
    u = make_uav(geofence_radius=100000.0, command_interval=100.0,
                 recenter_dist=1000.0)
    s1 = u.follow_target(CENTER, CENTER, (W, H))
    assert s1.startswith("orbit")
    # immediate re-call with same centre -> rate-limited, no new command
    n = len(u.master.mav.orbits)
    s2 = u.follow_target(CENTER, CENTER, (W, H))
    assert "holding" in s2 and len(u.master.mav.orbits) == n, (s2, n)
    # lost target clears the held centre
    u.notify_no_target()
    assert u._last_center is None
    print("rate-limit holds centre; lost-target clears it")


if __name__ == "__main__":
    test_geo()
    test_geo_from_range()
    test_safety_gates()
    test_rate_limit_and_lost()
    print("\nALL FLIGHT TESTS PASSED")
