# Pre-test checklist

Two tests, in order: (1) land on ArUco tag using `ibvs_perching` as-is, (2) perch on a branch using detection_pipeline + ibvs + the new UDP bridge into `ibvs_perching`. This list covers what's needed for **Test 1**; branch-specific items are called out separately at the end for later.

Already fixed in this pass (no input needed, pure doc/consistency bugs):
- `ibvs_perching/launch/ibvs_perching.launch` — corrected two comments that gave contradictory tag/plate dimensions (30cm marker/40cm plate vs. the actual 20cm marker/30cm plate confirmed in `models/ar_tag/model.sdf`).

---

## A. Camera (blocking — deferred per your call, but this is the #1 blocker)

- [ ] Wire up the DSJ-3079-HE (or whatever camera ends up mounted) as a ROS image source. `aruco_detector.py` subscribes to `camera/color/image_raw` + `camera/color/camera_info` under `$UAV_NAMESPACE` — nothing currently publishes these for real flight (`session.yml`'s camera pane is just a `realsense2_camera` placeholder pushed to shell history, never run).
- [ ] Standard fix: a `usb_cam` (or equivalent V4L2/UVC) launch file publishing those two topics.
- [ ] **Camera calibration is required, not optional**: `aruco_detector.py` needs a real `CameraInfo.K` matrix. Without it, `image_callback` returns early forever — no detection, ever, no matter how good the vision looks. Run `camera_calibration` (ROS) once the camera is physically mounted, save the resulting `camera_info` YAML, and load it via the `usb_cam`/camera driver launch args.
- [ ] Confirm: is DSJ-3079-HE the actual final camera choice for this test, or still open?

## B. Physical ArUco tag

- [ ] Print `ibvs_perching/models/ar_tag/materials/textures/aruco_id0.png` (DICT_4X4_50, id 0) at **exactly 20×20 cm** for the marker itself, mounted on/with roughly a 30×30 cm plate and a 5 cm quiet zone around the marker, matching `models/ar_tag/model.sdf`/`model.config`.
- [ ] Marker size matching matters only for `point.z` (depth/standoff regulation) — `point.x/point.y` centering stays pixel-exact even if size is slightly off (per README), so this isn't fully blocking but affects how well the vertical standoff behaves.
- [ ] Decide where the tag will be placed for the test (ground, elevated stand, etc.) and confirm it's flat and rigid enough not to move/flex during the approach.

## C. FCU / ArduPilot parameters

- [ ] **`GUID_OPTIONS`**: currently `8` in `UAV/parameters/uav_params.param` (raw-thrust mode, needed by the other LARICS MPC flight profile — left as-is per your call). `session.yml` runs `mavparam set GUID_OPTIONS 0` at runtime before launching `ibvs_controller.py`. **Verify this line actually executes and succeeds every session** (mavros must be fully up first) before arming — if it silently fails, the README's own flight-tested failure mode is "vehicle flies away at ~5.4 m/s, straight past 300 m."
- [ ] **No `FLTMODE` slot is currently mapped to `GUIDED_NOGPS` (mode 20)** in `uav_params.param` (`FLTMODE1-6` = STABILIZE/STABILIZE/LOITER/LOITER/RTL/RTL). The documented safety fallback ("flip the RC switch back to GUIDED_NOGPS to re-engage" / to hand control back) needs a real switch position for this. Either:
  - Assign one of the 6 `FLTMODE` slots to `GUIDED_NOGPS` (20) and re-flash/set on the FCU, or
  - Confirm the actual intended recovery procedure if not using the RC mode switch for this (e.g. disarm via RC only).
- [ ] Confirm `GUIDED_NOGPS` is actually enabled/selectable on this frame class + firmware build (nothing found blocking it, but not independently verified either).
- [ ] `FCU_URL` in `rw_setup.sh` is `/dev/ttyUSB_px4:921600` — a udev-alias device name. Confirm this udev rule exists on the ground station/companion computer, or update to the real device path (e.g. `/dev/ttyACM0`, `/dev/ttyUSB0`) before first run.

## D. RC / safety pilot engagement

- [ ] **No code currently binds a real RC switch/button to `ibvs/start`/`ibvs/stop`.** `rc_to_joy.py` only bridges RC → a `Joy` message; nothing subscribes to it to call those services. The only thing that calls them today is `keyboard_rc.py`, which is SITL-only. For this test you have two options:
  - Manually call `rosservice call /$UAV_NAMESPACE/ibvs/start` (and `/ibvs/stop`) from a ground-station terminal during the flight (pre-typed in `session.yml`, works fine for a first test), **or**
  - Write a small bridge node mapping a real RC channel to those service calls (more work, not needed for a first cautious test).
- [ ] `custom_config/rc_mapping.yaml` — the shipped channel mapping (`throttle:2, roll:0, pitch:1, yaw:3, rc_on:6, mode:5`) does **not** match the "tukan reference" mapping quoted in its own comment (`throttle:0, roll:2, pitch:3, yaw:1, ...`). **Verify against your actual transmitter's real channel order before flight** — a wrong mapping here could misinterpret sticks in the `Joy` bridge (used for the RC status/monitoring path, not the FCU's own RC input, but still worth getting right).
- [ ] Confirm a safety pilot is briefed on: manual takeoff in STABILIZE, when/how `ibvs/start` gets triggered, and that flipping the RC mode switch always regains manual control regardless of controller state.

## E. Tuning / gains

- [ ] `custom_config/ibvs_params_rw.yaml` has real, previously flight-tested numbers (not placeholders) — but they were tested on a different airframe/rig (per the thesis lineage), **not yet validated on this specific vehicle**. Treat `pid_xy`/`pid_z` gains, `max_tilt`, `max_body_rate`, `kp_att`, `kp_hover`/`kv_hover`, `align_tolerance`, `tag_timeout` as a starting point to retune during initial hover/hold tests, not as final.
- [ ] `target_z: -0.5` (0.5 m standoff above tag) — confirm this is the standoff you actually want for this test.

## F. Dry-run / bring-up sequence (recommended order, doesn't require new code)

1. Bench-test camera + ArUco detection alone (`aruco_detector.py` + the printed tag, camera on a desk) — confirm `ibvs/target_point` publishes with sane `x/y` values as you move the tag, and `ibvs/debug_image` shows correct detection, before ever putting it on the vehicle.
2. Verify `mavros/state` reports armed + `GUIDED_NOGPS` correctly and `GUID_OPTIONS` actually reads back as 0 after the session.yml runtime override, with props off.
3. Props-off "flight" test: arm, manually trigger `ibvs/start` with a fake/manual `ibvs/target_point` publish (there's already a pre-typed all-zero `rostopic pub --once ... geometry_msgs/PointStamped '{}'` in `session.yml` for exactly this), confirm `mavros/setpoint_raw/attitude` outputs sane values and the state machine transitions as expected, and confirm RC mode-switch takeover works, before ever spinning propellers.
4. Only then move to a tethered/low-altitude hover test with props, then the actual ArUco approach.

---

## Deferred to Test 2 (branch perching) — not needed for Test 1

Confirmed done (per your check):
- Detection pipeline + ibvs on the up-facing camera — confirmed tested.
- Physical gripper/perch mechanism — confirmed built.
- `udp_target_point_receiver.py` bridge — structurally verified against the old, confirmed-working `udp_receiver_node.py` from the thesis project: identical socket/timeout/shutdown pattern, only the payload (three `Float32` topics → one `PointStamped`) and JSON keys differ, and sender/receiver keys (`point_x`/`point_y`/`point_z`) match exactly. Still worth one live bench test (fake UDP packets → confirm `ibvs/target_point` comes out right in Docker) before flight, but no design risk left.

Still open:
- **`t_z` sign fix in `ibvs_controller.py`** — `target_callback` hardcodes `self.t_z = -msg.point.z` (target always *below* the vehicle, baked in for a floor-mounted tag under a down-facing camera). For the branch case (camera up, branch above) this needs to become `self.t_z = msg.point.z`, with `target_z` in the config changed from negative to positive (climb target above start, instead of standoff below). **Deliberately left unchanged for now** so Test 1 (ArUco, down-facing convention) is unaffected — apply only when starting Test 2.
- Axis mapping otherwise doesn't need new design work: `ibvs_perching` has no OptiTrack-specific frame anywhere — `compute_body_rates()` uses only the FCU's own AHRS attitude/velocity (`mavros/local_position/odom`) for body-FLU rates, and the target error comes from the vision point, not from any world-frame position. So beyond the `t_z` sign above, no other coordinate-system change is needed to drop OptiTrack.
- Perch-trigger mechanism — resolved by design, not by a triggered event: `PointController.depth_m` (constant pseudo-depth, default 0.3 m) combined with `ibvs_perching`'s `target_z` (kept further/less-reachable than depth_m) produces a small, steady, non-converging climb-rate command for the whole approach, since alignment and climb run in the same loop with no separate ALIGN/APPROACH states. Needs flight verification, not new design, once the `t_z` sign above is flipped for the branch case.
- **New: gripper-to-camera spatial offset.** The camera and the perch mechanism (gripper) are not at the same physical point on the airframe — there's a real, fixed 3D offset between where the camera looks and where the gripper actually contacts the branch. This offset needs to be measured (or taken from CAD) and added into the approach/perch geometry so the vehicle doesn't center the branch on the camera and then miss it with the gripper. Not yet accounted for anywhere in the code — needs a decision on where this correction is applied (e.g. as a constant bias on the commanded position, or folded into `target_x`/`target_y`/`target_z`).
- Nicla Vision camera mount/orientation re-check (`cv2.rotate(..., ROTATE_180)` in `NiclaSource.py`) for the new upward mount — depends on the physical camera orientation once mounted, can't be resolved from code alone.
