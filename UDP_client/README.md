# UDP Client

Runs the full perching stack (detection → IBVS → control) and sends pixel error and ToF distance to the flight controller over UDP each frame. Also records raw video to disk.

---

## What It Does

`main_record.py` is the top-level entry point for the complete system:

1. Builds a `DetectionPipeline` (NiclaSource + HailoSegDetector or MP4 + YOLO)
2. Wraps it in an IBVS pipeline (KLT tracking + PointController)
3. Each frame: writes to MP4, sends UDP packet, prints verbose status

---

## UDP Packet Format

JSON sent to `192.168.1.90:5005` each frame that has a valid control error:

```json
{
  "error_x":    -12.4,
  "error_y":    3.1,
  "tof":        480.0,
  "timestamp":  1748000000.0
}
```

| Field | Description |
|-------|-------------|
| `error_x` | Pixel distance of branch point from frame center (positive = right) |
| `error_y` | Pixel distance from frame center (positive = down) |
| `tof` | ToF reading from Nicla Vision in mm, `-1.0` if unavailable |
| `timestamp` | Unix timestamp |

The Docker-side ROS node (`udp_receiver_node.py`) receives this and publishes to `/perch/error_x`, `/perch/error_y`, `/perch/tof`.

---

## Source Selection

Set `USE_NICLA` at the top of `main_record.py`:

```python
USE_NICLA = True   # Nicla Vision camera + Hailo-8 .hef model
USE_NICLA = False  # MP4 file + YOLO .pt model
```

---

## Directory Structure

```
UDP_client/
├── main_record.py       # Full pipeline entry point with recording + UDP send
├── main.py              # Minimal entry point (no recording)
├── client/
│   └── udp_client.py    # UDPSender — wraps socket, sends JSON
└── recordings/          # MP4 recordings saved here (timestamped)
```

---

## Running

```bash
cd UDP_client
python3 main_record.py
```

Recordings are saved to `recordings/YYYYMMDD_HHMMSS_raw.mp4`.

Press `q` in the display window (if `DISPLAY` is set) to stop, or `Ctrl+C`.

---

## Receiving (Docker / ROS side)

The perch mission launch file starts the receiver automatically:

```bash
roslaunch pearch_mission perch_mission.launch
```

Or manually:

```bash
rosrun pearch_mission udp_receiver_node.py
```

---

## Dependencies

```bash
pip install opencv-python numpy pyserial

# Plus all detection_pipeline and ibvs dependencies
pip install ultralytics supervision scipy scikit-image pyyaml hailo_platform
```
