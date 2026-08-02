# Autonomous_perching

Competition submission: an autonomous UAV branch-perching system, adapted from
the [Diplomski_rad](https://github.com/YassinBaraa/Diplomski_rad) master's
thesis project.

---

## Overview

Unlike the thesis setup, the competition scenario starts with the UAV already
positioned under the branch. There is no OptiTrack localization and the ToF
sensor is not used for the perch maneuver itself (it may be repurposed for
something else later). The camera faces **up** instead of forward.

The perception stack (branch detection, candidate point selection, KLT
tracking) is reused largely as-is from the thesis — it's pure image-space
code and doesn't depend on camera orientation. The flight-side ROS package is
new: instead of the thesis's OptiTrack-dependent, MAVROS-tracker-based perch
controller, this project uses
[`ibvs_perching`](https://github.com/JakobDomislovic/ibvs_perching) — a
minimal IBVS controller that commands ArduPilot directly (body rates +
climb rate via `mavros/setpoint_raw/attitude`), with no position tracker and
no motion capture dependency.

**Current milestone: `ibvs_perching` is being run as-is, with its stock
AprilTag/ArUco vision module, as a first end-to-end test of the perching
maneuver and the direct-attitude-control approach.** The next step is
swapping the AprilTag detector for this project's own `detection_pipeline` +
`ibvs` output (branch candidate point instead of a tag), feeding
`ibvs_perching`'s `ibvs/target_point` interface.

---

## Repository Structure

This is the root repository. Subdirectories are git submodules (except
`UDP_client`, a plain folder).

```
Autonomous_perching/
├── detection_pipeline/   # YOLOv8-seg + Hailo + branch candidate point pipeline (from thesis)
├── ibvs/                 # KLT feature tracking + proportional visual controller (from thesis)
├── ibvs_perching/        # ROS IBVS flight controller (AprilTag test rig; MAVROS body-rate control)
├── UDP_client/           # Entry point, recording, UDP send (from thesis)
├── model_training/       # YOLOv8 training, SAM annotation, Hailo export (from thesis)
└── UAV/                  # ROS flight_setup package, Pixhawk params, CAD (from thesis)
```

## Submodules

| Repo | Branch | Purpose |
|------|--------|---------|
| [detection_pipeline](detection_pipeline/) | `rektor_modification` | Branch detection and perch point estimation |
| [ibvs](ibvs/) | `rektor_modification` | Visual servoing — KLT tracking + PointController |
| [ibvs_perching](ibvs_perching/) | `master` (upstream) | ROS IBVS flight controller — direct MAVROS attitude control, currently AprilTag-based |
| [model_training](model_training/) | `rektor_modification` | YOLOv8 training and Hailo model compilation |
| [UAV](UAV/) | `rektor_modification` | Flight controller ROS package, PX4 parameters, CAD |

`ibvs_perching` is a plain (unforked) clone of the upstream repo — it isn't
modified yet, so it stays on its own `master` branch rather than
`rektor_modification`.

---

## Cloning

```bash
git clone --recurse-submodules git@github.com:YassinBaraa/Autonomous_perching.git
# or after cloning:
git submodule update --init --recursive
```

---

## What still needs to change from the thesis version

- **`ibvs_perching`'s vision interface** assumes a down-facing camera
  (`image right = body forward`). Needs a new adapter node that publishes
  `ibvs/target_point` from `detection_pipeline`/`ibvs`'s branch candidate
  point, with axis conventions re-derived for the up-facing camera.
- **`detection_pipeline/sources/NiclaSource.py`** hardcodes a 180° image
  rotation for the old forward-facing mount — re-check against the new
  up-facing mount.
- **Hardcoded IPs** in `UDP_client` and `UAV/flight_setup/startup/*`
  reference the old network setup (including the now-unused OptiTrack
  server) — reconcile once the new setup's topology is finalized.
- **Model weight files** (`.pt`/`.onnx`) — some exceed GitHub's file size
  limits and are not committed to `model_training`; they're expected to be
  placed directly in the project root instead.

---

## Platform

| Component | Details |
|-----------|---------|
| Compute | Raspberry Pi 5 |
| AI accelerator | Hailo-8 AI HAT (hailort 4.23) |
| Camera + ToF | Arduino Nicla Vision (OV5647 + VL53L1X), facing up |
| Flight controller | Pixhawk (MAVROS / ArduPilot) |
| Localization | None (no OptiTrack) |
| ROS environment | ROS Noetic |
