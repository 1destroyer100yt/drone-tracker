--[[
   Optional on-vehicle guard for the tracker UAV.

   Runs on the autopilot (needs AP_SCRIPTING_ENABLED, which this custom build
   turns on). It watches for the companion computer's follow commands: if the
   plane is in GUIDED but hasn't been repositioned for a while -- i.e. the
   Raspberry Pi / tracker link has gone quiet -- it just warns the operator.

   It deliberately does NOT take control (no mode change, no RTL). Recovery is
   left to the pilot and the autopilot's own failsafes, matching the
   camera-only / passive safety stance of uav.py. Extend as needed.

   Install: copy to the flight controller's APM/scripts/ folder, set
   SCR_ENABLE=1, reboot.
--]]

local TIMEOUT_MS = 5000        -- warn if no guided update within this window
local last_reposition_ms = millis()
local last_lat, last_lng = 0, 0

function update()
   -- only relevant while flying a guided mission (the follow mode)
   if not arming:is_armed() or vehicle:get_mode() ~= 15 then  -- 15 = GUIDED (Plane)
      last_reposition_ms = millis()
      return update, 1000
   end

   local target = vehicle:get_target_location()
   if target then
      local lat, lng = target:lat(), target:lng()
      if lat ~= last_lat or lng ~= last_lng then
         last_lat, last_lng = lat, lng
         last_reposition_ms = millis()          -- companion is still steering
      end
   end

   if millis() - last_reposition_ms > TIMEOUT_MS then
      gcs:send_text(4, "tracker: no follow update >5s (companion link?)")
      last_reposition_ms = millis()             -- rate-limit the warning
   end

   return update, 500
end

gcs:send_text(6, "tracker_failsafe.lua loaded")
return update, 1000
