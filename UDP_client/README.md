# UDP Client

Runs the full perching stack (camera → detection → IBVS) and sends the perch/land point in pixel coordinates over UDP each frame. `main_record.py` also records raw video to disk; `main.py` is the same pipeline without recording.

---

## What It Does

Both entry points call `pipeline_factory.build_pipeline()`, which:

1. Builds a raw camera source (DSJ / Nicla / Pi camera / MP4 — see below) — this is independent of detection mode; ArUco is just a detector choice, not a camera choice
2. In `"branch"` mode, wraps it in the segmentation `DetectionPipeline`; in `"aruco"` mode, wraps it directly in `ArucoSource`
3. Wraps whichever source in the IBVS pipeline (KLT tracking + PointController for local visualization)

Each frame, whichever entry point you run sends a UDP packet with the current target point (KLT-tracked post-lock, or the raw detection/ArUco point pre-lock).

---

## UDP Packet Format

JSON sent to `192.168.0.128:5005` (see `UDPSender` in `client/udp_client.py`) each frame that has a target point:

```json
{"px": 412.3, "py": 198.7}
```

`px`/`py` are the perch/land point in pixel coordinates (image space, not offset from center). Nothing else is sent — no ToF/distance, it isn't used anywhere in the stack anymore.

> **Receiving side note:** this repo has no UDP receiver for `ibvs_perching` (the current MAVROS/docker package) — it expects an `ibvs/target_point` `PointStamped` published directly in ROS (see `ibvs_perching/scripts/aruco_detector.py`). Bridging this UDP packet into `ibvs/target_point` would need a small ROS node on the docker side — out of scope here unless you want it built.

---

## Source & Detection Mode Selection

Both `main.py` and `main_record.py` share the same switch — set these at the top of `pipeline_factory.py`:

```python
# Camera — independent of detection mode
SOURCE_TYPE = "dsj"      # DSJ-3079-HE USB camera (default)
SOURCE_TYPE = "nicla"    # Nicla Vision camera over USB serial
SOURCE_TYPE = "camera"   # Raspberry Pi camera via picamera2
SOURCE_TYPE = "mp4"      # video file, for offline testing

# Detector
DETECTION_MODE = "branch"  # branch segmentation pipeline -> final_point
DETECTION_MODE = "aruco"   # direct ArUco marker detection -> ibvs/sources/ArucoSource.py
```

`ArucoSource` detects any marker from the configured dictionary (`ARUCO_DICTIONARY`, default `"auto"` — tries every predefined dictionary during warmup and uses whichever finds the tag, since there's no way to know which family a given printed/generated marker uses; set it to a specific name like `"DICT_4X4_50"` once you know yours, to skip the scan) — it doesn't filter by marker ID either, since this is a single-tag perch/land setup, not multi-tag identification. It also requires **opencv-contrib-python** (`cv2.aruco`) — plain `opencv-python` does not include it.

---

## Directory Structure

```
UDP_client/
├── pipeline_factory.py  # Shared camera + detection-mode selection, used by both entry points below
├── main_record.py       # Full pipeline entry point with recording + UDP send
├── main.py              # Same pipeline, no recording
├── client/
│   └── udp_client.py    # UDPSender — wraps socket, sends JSON
└── recordings/          # MP4 recordings saved here (timestamped, main_record.py only)
```

---

## Running

```bash
cd UDP_client
python3 main_record.py   # with recording + local display (if DISPLAY is set)
python3 main.py          # minimal, UDP send only
```

`main_record.py` saves one timestamped run to up to three files in `recordings/`:

| File | Contents |
|---|---|
| `YYYYMMDD_HHMMSS_raw.mp4` | Unannotated camera frames |
| `YYYYMMDD_HHMMSS_ibvs_overlay.mp4` | The "IBVS" window — tracked features, target point, crosshair, velocity arrow |
| `YYYYMMDD_HHMMSS_detection_overlay.mp4` | The "Detection Pipeline" window — branch segmentation debug overlay (`"branch"` mode only) |

All three are written whether or not a display is attached (`HAS_DISPLAY` only gates the live `cv2.imshow` windows, not recording).

Press `q` in the display window (if `DISPLAY` is set) to stop, or `Ctrl+C`.

---

## Dependencies

```bash
pip install opencv-contrib-python numpy pyserial

# Plus all detection_pipeline and ibvs dependencies
pip install ultralytics supervision scipy scikit-image pyyaml hailo_platform
```
