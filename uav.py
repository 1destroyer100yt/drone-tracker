"""ArduPilot / MAVLink link for the body & face tracker.

Two levels of aircraft interaction, both opt-in and both fenced by safety
checks:

  1. GIMBAL AIM (default) -- MOUNT_CONTROL points a servo camera mount at the
     tracked target. Camera moves; the aircraft does not.

  2. FOLLOW / ADVANCED FLIGHT (--follow) -- estimates the target's position on
     the ground from the camera line-of-sight and commands the plane to ORBIT
     it at a standoff radius, updating the orbit centre as the person moves.
     A fixed-wing can't hover, so orbiting a subject is the correct follow
     behaviour (this is how filming/loiter planes work).

Safety model for FOLLOW (deliberately conservative):
  - Acts ONLY when the aircraft is already ARMED and in GUIDED mode. It never
    arms, never changes mode, never overrides RC. Flip out of GUIDED (or take
    the sticks) and the plane stops obeying us instantly.
  - Enforces a minimum altitude and a standoff orbit radius, so it circles the
    subject at a distance -- it never flies AT a person.
  - Refuses to command if the estimated target is outside a geofence radius,
    if telemetry is stale, or if the line-of-sight is too shallow to locate
    the ground reliably.
  - On lost target it stops sending: the plane simply holds its current guided
    orbit. Recovery of the airframe (RTL etc.) is left to the pilot and the
    autopilot's own failsafes.

  These estimates assume flat ground and a body-fixed (strapdown) camera.
  VALIDATE IN ArduPilot SITL before ever flying this. See test_flight.py.

Connection strings (pymavlink style):
  udpout:127.0.0.1:14550   -> SITL / a GCS forwarding port (typical testing)
  /dev/ttyAMA0             -> Pi serial to a Pixhawk (add ,57600 baud below)
"""

import math
import time

from pymavlink import mavutil

import geo


class FollowConfig:
    """Tunables for --follow. Metres, seconds, degrees where noted."""

    def __init__(self, orbit_radius=80.0, orbit_speed=15.0, min_alt=30.0,
                 geofence_radius=300.0, cam_tilt_deg=0.0,
                 telemetry_timeout=2.0, command_interval=1.0,
                 recenter_dist=8.0):
        self.orbit_radius = orbit_radius        # standoff distance (m)
        self.orbit_speed = orbit_speed          # tangential speed (m/s)
        self.min_alt = min_alt                  # refuse below this AGL (m)
        self.geofence_radius = geofence_radius  # max target dist from home (m)
        self.cam_tilt = math.radians(cam_tilt_deg)  # fixed camera down-tilt
        self.telemetry_timeout = telemetry_timeout
        self.command_interval = command_interval    # min s between orbit cmds
        self.recenter_dist = recenter_dist          # resend if centre moved > m


class MavlinkUAV:
    def __init__(self, connection, hfov_deg=62.2, vfov_deg=None,
                 baud=57600, source_system=1, follow_config=None):
        self.hfov = math.radians(hfov_deg)
        self.vfov = math.radians(vfov_deg) if vfov_deg else None
        self.master = mavutil.mavlink_connection(
            connection, baud=baud, source_system=source_system)
        self.connected = False
        self.cfg = follow_config or FollowConfig()

        # telemetry state, filled by update_from_telemetry()
        self.armed = False
        self.mode = None
        self.lat = self.lon = None          # deg
        self.rel_alt = None                 # m above home
        self.yaw = self.pitch = None        # rad
        self.home = None                    # (lat, lon)
        self.last_telemetry = 0.0

        # follow command rate-limiting
        self._last_cmd_t = 0.0
        self._last_center = None

    # ----- link setup -------------------------------------------------------
    def wait_heartbeat(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            hb = self.master.wait_heartbeat(timeout=1.0)
            if hb is not None:
                self.connected = True
                return True
        return False

    def request_streams(self, rate_hz=10):
        """Ask the autopilot to send attitude + position at rate_hz."""
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, rate_hz, 1)
        except Exception:
            pass

    def update_from_telemetry(self):
        """Drain pending MAVLink messages and refresh cached vehicle state.
        Non-blocking; call once per frame."""
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                break
            kind = msg.get_type()
            if kind == "HEARTBEAT":
                self.armed = bool(msg.base_mode &
                                  mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = self.master.flightmode
                self.last_telemetry = time.monotonic()
            elif kind == "GLOBAL_POSITION_INT":
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.rel_alt = msg.relative_alt / 1000.0
                self.last_telemetry = time.monotonic()
            elif kind == "ATTITUDE":
                self.yaw = msg.yaw
                self.pitch = msg.pitch
            elif kind == "HOME_POSITION":
                self.home = (msg.latitude / 1e7, msg.longitude / 1e7)

    # ----- pixel -> angle ---------------------------------------------------
    def _angles(self, target_xy, center_xy, frame_wh):
        """Pixel target -> (angle_x, angle_y) rad in the body frame:
        +x right, +y down, zero at boresight."""
        tx, ty = target_xy
        cx, cy = center_xy
        w, h = frame_wh
        vfov = self.vfov if self.vfov else self.hfov * (h / w)
        angle_x = (tx - cx) / (w / 2.0) * (self.hfov / 2.0)
        angle_y = (ty - cy) / (h / 2.0) * (vfov / 2.0)
        return angle_x, angle_y

    # ----- gimbal (camera only) --------------------------------------------
    def send_gimbal(self, target_xy, center_xy, frame_wh):
        """Point the servo gimbal at the target. Camera moves; aircraft doesn't."""
        angle_x, angle_y = self._angles(target_xy, center_xy, frame_wh)
        self.master.mav.mount_control_send(
            self.master.target_system,
            self.master.target_component,
            int(math.degrees(-angle_y) * 100),   # pitch  (centi-deg)
            0,                                    # roll
            int(math.degrees(angle_x) * 100),    # yaw    (centi-deg)
            0,                                    # save position flag
        )

    # ----- follow / advanced flight ----------------------------------------
    def estimate_target(self, target_xy, center_xy, frame_wh, range_m=None):
        """Estimate the target's ground position from the camera LOS and the
        current aircraft telemetry. If range_m (a measured slant distance to the
        person, e.g. from distance.py) is given, use it instead of the
        flat-ground assumption. Returns (lat, lon, horiz_range) or None."""
        if None in (self.lat, self.lon, self.rel_alt, self.yaw, self.pitch):
            return None
        angle_x, angle_y = self._angles(target_xy, center_xy, frame_wh)
        # body-fixed camera: LOS off-axis = pixel angle (+ any fixed down-tilt)
        heading, depression = geo.line_of_sight(
            self.yaw, self.pitch,
            gimbal_yaw=angle_x,
            gimbal_pitch=self.cfg.cam_tilt + angle_y)
        if range_m and range_m > 0:
            return geo.project_from_range(self.lat, self.lon, heading,
                                          depression, range_m)
        return geo.project_target(self.lat, self.lon, self.rel_alt,
                                  heading, depression)

    def follow_target(self, target_xy, center_xy, frame_wh, range_m=None):
        """Command an orbit around the estimated target position, subject to
        all safety gates. range_m is an optional measured distance to the person
        (from distance.py) that improves the target estimate. Returns a short
        status string for logging."""
        cfg = self.cfg
        now = time.monotonic()

        if now - self.last_telemetry > cfg.telemetry_timeout:
            return "no telemetry"
        if not self.armed:
            return "disarmed"
        if self.mode != "GUIDED":
            return f"not GUIDED ({self.mode})"
        if self.rel_alt is None or self.rel_alt < cfg.min_alt:
            return f"below min alt ({self.rel_alt})"

        est = self.estimate_target(target_xy, center_xy, frame_wh, range_m)
        if est is None:
            return "LOS too shallow / no fix"
        tlat, tlon, _ = est

        home = self.home or (self.lat, self.lon)
        if geo.haversine_m(home[0], home[1], tlat, tlon) > cfg.geofence_radius:
            return "target outside geofence"

        # rate-limit: only resend when enough time passed or centre moved
        moved = (self._last_center is None or
                 geo.haversine_m(self._last_center[0], self._last_center[1],
                                 tlat, tlon) > cfg.recenter_dist)
        if now - self._last_cmd_t < cfg.command_interval and not moved:
            return "orbiting (holding centre)"

        self._send_orbit(tlat, tlon, self.rel_alt)
        self._last_cmd_t = now
        self._last_center = (tlat, tlon)
        return f"orbit r={cfg.orbit_radius:.0f}m @ {tlat:.6f},{tlon:.6f}"

    def _send_orbit(self, lat, lon, alt_rel):
        """Command the plane to loiter around a global point at the standoff
        radius. Uses MAV_CMD_DO_REPOSITION (the ArduPilot Plane guided
        'go loiter here') -- in GUIDED the aircraft circles this point at the
        given radius, which is exactly the orbit-follow behaviour we want."""
        self.master.mav.command_int_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            mavutil.mavlink.MAV_CMD_DO_REPOSITION,
            0, 0,                                  # current, autocontinue
            self.cfg.orbit_speed,                  # p1 ground speed (m/s)
            0,                                     # p2 bitmask (0)
            self.cfg.orbit_radius,                 # p3 loiter radius (m, Plane)
            float("nan"),                          # p4 yaw (unchanged)
            int(lat * 1e7), int(lon * 1e7),        # x, y
            alt_rel,                               # z (relative alt)
        )

    def notify_no_target(self):
        """Lost target: stop commanding. The plane holds its current guided
        orbit; airframe recovery is the pilot's / autopilot's job."""
        self._last_center = None

    def close(self):
        try:
            self.master.close()
        except Exception:
            pass
