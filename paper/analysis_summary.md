# Vision-based branch perching — analysis for the paper

All numbers here come from the files in `paper/data/`, and every figure is in `paper/figures/` as both PDF (vector, fonts embedded) and 300 dpi PNG.

- **Figure format:** sized for an IEEE column, 3.5 in wide, or 7.16 in for the two-panel-wide ones. Text is Times-like at 8 pt.
- **Script names:** each section names the script that makes its numbers (in `paper/scripts/`).
- **Source data:** the flight logs and videos in `UDP_client/recordings/` (10-12 and 10-13), and the drone's ROS bags.

---

## 0. Data the analysis is built on

| Source | What | Amount |
|---|---|---|
| Vision recordings (`UDP_client/recordings/run_*.log` + videos) | every frame: state, point, size, events, per-stage timing | 35 runs, 33 videos, 23 min of tracking, 33 581 tracked + 4 574 detector frames |
| ROS bags (`real_world/` and `optitrack/` in the container) | controller state, target point, size, closeness/TTC, IMU, motion capture (≈224 Hz), battery, ESC | 39 bags exported to CSV (`data/bags/`, 229 MB) |
| Detector survey (`detect_survey.py`, on the Hailo) | every 20th frame of all 33 videos re-run through the detector with per-step timing | 1 510 frames, 1 703 detections |
| Resource run (`resource_run.py`, on the Hailo) | CPU, memory, temperature, SoC power while replaying a flight | idle 20 s + 1 flight replay + 30 s detector only |

The runs were paired with the bags by wall-clock overlap, which matched 34 of the 35. Both were recorded on the same machine clock, so the vision events and the drone's IMU and mocap line up in time.

---

## 1. System

| | |
|---|---|
| Computer | Raspberry Pi 5 (4× Cortex-A76, 8 GB) |
| Accelerator | Hailo-8 (AI HAT), PCIe gen 2 ×1 |
| Camera | DSJ-3079-HE USB camera, 1280×720 |
| Network | YOLOv8m-seg, 1 class ("branch"), 27.2 M parameters, 104.7 GFLOPs at 640², compiled to a 3-context HEF (38.8 MB) |
| Network outputs | boxes (DFL, 16 bins) + class + 32 mask coefficients at strides 8/16/32; 160×160×32 prototype masks |
| Hailo benchmark (`hailortcli benchmark`) | **13.5 FPS, 54 ms** hardware latency (`data/hailo_benchmark.json`) |

**About the model metrics.** The repository has training records for the YOLO26 `.pt` checkpoints (`model/best_small.pt`: mask mAP50 = 0.616, box mAP50 = 0.706). **The network that flies is the YOLOv8m-seg HEF**, and its validation numbers are not in the repo. For the paper you need the mAP of the deployed model on your validation set, which only exists on your training machine (`data/model_info.json` lists what was found).

---

## 2. Speed

Data comes from `data/timing_summary.json`. The script is `fig_timing.py`.

### Frame rate — `fig_frame_period`
The pipeline has two modes:

| Mode | What runs | Frame period median (p90 / p99) | Rate |
|---|---|---|---|
| Searching | Hailo detector + mask post-processing | **165 ms** (226 / 335) | 6.1 Hz |
| Tracking | KLT optical flow + branch-width measurement | **37 ms** (77 / 120) | 27 Hz |

The detector runs only until a perch point is locked. After that, the point is tracked at the camera frame rate and the network is no longer in the loop.

### Where a detection frame's time goes — `fig_detection_time`
This comes from the detector survey: medians, with the 90th percentile as whiskers.

| Step | ms |
|---|---|
| colour + resize to 640² | 2.3 |
| **Hailo-8 inference** | **51.9** |
| de-quantise outputs | 4.2 |
| decode boxes + NMS | 10.3 |
| assemble masks | 2.1 |
| merge masks | 2.3 |
| distance transform | 3.9 |
| **skeletonise** | **26.1** |
| choose point (scoring) | 15.0 |
| debug image (visualiser) | 10.5 |
| **sum** | **128.6** |

The accelerator takes about 40% of a detection frame and the host takes the rest. The two cheapest wins:
- dropping the debug image in flight (−10 ms);
- skeletonising a smaller crop (the skeleton is the largest host step).

The in-flight log split agrees: detect 77 ms + post-process 53 ms.

### Latency — `fig_latency`
| Path | Median | p90 | p99 |
|---|---|---|---|
| camera grab → UDP packet sent (vision side) | **22.7 ms** | 63.9 | 117.7 |
| UDP packet → attitude command (controller side) | 19.2 ms | 30.0 | – |
| **camera grab → attitude command** | **48.0 ms** | 87.2 | 142.3 |

The grab time travels in every packet (`age`), and the drone's `/red/ibvs/target_point` stamp is the grab time. That makes the end-to-end number a direct measurement, not a sum of guesses. The main loop's own overhead is 2 ms median (6 ms p99).

---

## 3. Memory, CPU, power, temperature

Data comes from `data/resources_summary.json`. The scripts are `resource_run.py` and `fig_resources.py`, and the figure is `fig_resources`.

| Mode | Process CPU¹ | CPU time / frame | Memory (RSS) | SoC power² | SoC temp |
|---|---|---|---|---|---|
| idle (no vision) | – | – | – | 1.80 W | 64.5 °C |
| searching (full pipeline) | 92 % | 172 ms | **258 MB** (max 275) | 2.91 W | 71 °C |
| tracking (full pipeline) | (221 %)³ | **48.6 ms** | 252 MB | (4.86 W)³ | 71 °C |
| detector + post-processing only | 78 % | 113 ms | 203 MB | 3.50 W | 73 °C |

¹ 100 % = one of the four cores.

² The sum of the Pi 5 PMIC rails (SoC, RAM, I/O). It does **not** include the Hailo HAT or the USB camera.

³ In the replay, tracking frames came every 15 ms because nothing waited for the camera. In flight they come every 37 ms, so CPU % and power here are higher than in flight. **CPU time per frame** doesn't depend on that, so report that number.

What to say in the paper:
- **Memory:** the whole vision process stays under 0.3 GB, about 3% of the 8 GB board.
- **CPU load:** the CPU is never saturated. Searching uses about one core; tracking needs about 49 ms of CPU per 37 ms frame, about 1.3 cores, spread over threads.
- **Power and temperature:** about 1–3 W above idle, and temperature steady at 71–73 °C.

---

## 4. Detection quality

Data comes from `data/detection_summary.json`, `data/detection/*.csv`. The scripts are `detect_survey.py`, `mask_vs_edge.py` and `fig_detection.py`.

### What gets detected — `fig_detection_conf`
| Branch angle in the image | Detections | Median confidence |
|---|---|---|
| vertical (≥ 60°) | 1 426 | 0.875 |
| diagonal | 161 | 0.878 |
| horizontal (≤ 20°) | 116 | 0.751 |

The horizontal detections are mostly the ceiling beam: a real object that looks like a branch. Under the old orientation rule it could be chosen. The **vertical-only rule (±30°)** removes it.
- **Frames where the old rule picked a non-vertical branch:** 4.0%.
- **Frames with a valid candidate:** 97.5% under the old rule, 95.8% under the new one.

So the restriction costs about 2% of candidates.

### Mask width vs. true width — `fig_mask_width`
On 1 216 points, the branch width was compared two ways:
- across the network's mask;
- between the image edges on the same cross-section.

The **mask is 2.2× wider** than the edges (median edge/mask = 0.45, IQR 0.39–0.54). The reason is the 160×160 prototype resolution: one mask cell is 8 px of the 1280-px frame, and that dominates for thin branches. That is why the size sent to the drone is measured from image edges (`BranchWidthTracker`), not from the mask.

### Two decoder issues — `fig_decoder_issues`
Neither issue is fixed in the deployed code.
1. **Box offset.** Box centres are missing YOLOv8's +0.5-cell anchor offset. Masks are clipped to their box, so they lose pixels: with a correctly placed box they would keep **+20.5%** (median) more pixels, and 86% of detections would gain more than 10%.
2. **NMS box format.** `cv2.dnn.NMSBoxes` receives x1,y1,x2,y2 where it expects x,y,w,h. Fixing it changes the result in only **1%** of frames (12 detections in total).

Fix 1 before writing "mask quality" numbers in the paper. Fix 2 has negligible impact. Both are fixed in the working tree since 2026-10-14 (not yet flown, not committed); see the docs page `docs/branch_detection_pipeline.html`, section "Validation".

### Input preparation: squash vs. letterbox — `fig_input_modes`
Data comes from `data/input_modes_summary.json` and `data/detection/input_modes.csv`. The script is `input_modes.py` (it uses the Hailo), and the figure script is `fig_input_modes.py`.

The flown code squashes 1280×720 to 640×640 (x × 0.5, y × 0.89). The test sent 746 recorded frames to the network four ways, all with the corrected decoder, and compared them inside the central 640×640 region that all four see:

| Input | Frames with detection | Median max. confidence | Frames with a candidate | … at threshold 0.8 | Mask width error |
|---|---|---|---|---|---|
| squash (as flown) | 86.1 % | 0.886 | 80.2 % | 58.4 % | 12.8 px |
| **letterbox** | **93.0 %** | **0.929** | **89.3 %** | **81.6 %** | 12.1 px |
| centre crop 720² → 640² | 94.6 % | 0.922 | 88.6 % | 84.3 % | 8.0 px |
| centre crop 640² (full res.) | 94.1 % | 0.914 | 92.4 % | 86.7 % | 7.0 px |

What this shows:
- **Letterbox is the clear choice.** It raises confidence in 84% of frames (median +0.05) and the share of frames with a candidate by 9 points, with the same field of view, and it is faster to prepare (0.8 vs 1.4 ms). This strongly suggests the network was trained on letterboxed images.
- **A higher threshold becomes affordable.** With letterbox, a confidence threshold of 0.8 still gives more candidates than squash at 0.5.
- **Crops help the masks, not the coverage.** A full-resolution centre crop halves the mask width error (4 px per prototype cell instead of 8), but sees only half the image width.

### Skeleton neighbour count (found 2026-10-14)
`CandidateScoring` counted skeleton neighbours with `cv2.filter2D` and a `uint8` kernel. In OpenCV 4.10 that returns wrong sums (1 along a line instead of 3), so end points and junctions were almost never found. In every flight so far, the end-point rule (≥ 5 widths from a branch tip) and the junction part of the crossing rule were therefore effectively off. The fix is a float32 kernel. Do not describe these two rules as active for the flights analysed here.

---

## 5. Tracking

Data comes from `data/tracking_summary.json`, `data/runs_metrics.csv`. The script is `fig_tracking.py`.

### Continuity — `fig_tracking` (left)
Over 35 flights, with a median over runs:
- a **perch point was sent on 87.6%** of frames;
- packets carried a **size in 96.4%** of cases.

There are 5.5 tracking losses per minute of tracking time.

### Re-finding a lost target — `fig_tracking` (middle)
| | |
|---|---|
| losses with a re-find | 106 |
| re-found **in the same frame** as the loss | **37 %** |
| median time to re-find | **0.46 s** |
| re-found within 1 s | 62 % |
| 90th percentile | 10.3 s |

The long tail (10% above 10 s) was not analysed case by case.

### Put-back corrections — `fig_tracking` (right)
When the tracked point drifts away from the stored target, it is put back. Over 206 put-backs:
- the correction is **22.5 px** median and 80 px at the 90th percentile;
- each was confirmed by a median of 15 ORB matches.

### Size measurement noise — `fig_size_noise`
Noise is the residual of ln(size) from a straight line over 9 consecutive frames. That removes the true growth while approaching, which a plain frame-to-frame difference would count as noise.

| Apparent width | Median | p90 | Median in pixels |
|---|---|---|---|
| < 8 px | 1.5 % | 5.1 % | 0.11 px |
| 8–15 px | 2.3 % | 6.0 % | 0.23 px |
| 15–40 px | 1.4 % | 4.1 % | 0.30 px |
| > 40 px | 0.5 % | 3.9 % | 0.35 px |
| **all** | **1.8 %** | **5.4 %** | **0.17 px** |

The width is measured to **sub-pixel precision**, about 0.2 px. Below about 6 px, noise rises to 7% (the measurement limit).

### What makes the tracker lose the target — `fig_loss_vs_rate`
Tracked time was cut into 0.5 s windows (2 835 windows). Each window's peak body rate |ω| comes from the IMU, and the plot shows how often a loss happened in it.

| Max \|ω\| in window | Windows | P(loss) |
|---|---|---|
| < 10 °/s | 2 089 | **0.7 %** |
| 10–20 °/s | 233 | 1.7 % |
| 20–30 °/s | 146 | 6.2 % |
| 30–45 °/s | 171 | 9.9 % |
| 45–60 °/s | 82 | 19.5 % |
| 60–90 °/s | 72 | **40 %** |
| > 90 °/s | 42 | **52 %** |

The same curve holds when only windows **far from the branch** (width < 15 px) are kept: 42% at 60–90°/s. So it is the motion itself, not being close to the branch.

This is the main robustness result: **tracking fails because of fast rotation (motion blur and large image shift), not because of the scene.** It also explains the bad 17:43–17:49 flights, which had 4–5× more camera shake.

---

## 6. Size as a distance sensor (approach precision)

Data comes from `data/approach_summary.json`, `data/approach_climbs.csv`. The script is `fig_approach.py`.

The apparent width follows **s = K/d**. This needs no camera calibration and no knowledge of the branch diameter. Ground truth is motion capture.

### Size vs. distance — `fig_size_distance`
| | |
|---|---|
| climbs with contact and a clean fit | 16 |
| R² of 1/s vs. height, per climb | **median 0.993** (min 0.940) |
| K (px·m), per climb | median 15.1 (13.6–22.1: different branches) |
| **distance from size, after the per-climb K** | **2.5 % median error**, 7.6 % p90 |
| camera-to-branch offset at contact | 7.9 cm (IQR 6.0–10.4): the camera sits about 12 cm from the gripper |

The size saturates below about 0.2 m: the branch fills the measurement window and the camera can't focus.

### Time-to-contact — `fig_ttc`
TTC comes from the size alone, as 1 / (d ln s / dt) over a 1 s least-squares window, the same as the controller. It is compared with the true TTC from mocap, over 2 449 samples from 16 climbs.

| True TTC | Median abs. error | p90 |
|---|---|---|
| 0–1 s | 19.9 % | 37 % |
| 1–2 s | 10.8 % | 24 % |
| **2–4 s** | **6.7 %** | 20 % |
| **4–8 s** | **7.5 %** | 26 % |
| 8–20 s | 20.5 % | 70 % |
| **all** | **10.4 %** | |

The error is lowest in the 2–8 s range, which is where the controller uses it. Far away the size barely changes; very close, it saturates.

### Example — `fig_approach_example`
One climb: height from mocap and the size sent by the vision, over the same time axis.

---

## 7. Height held from size only

Data comes from `data/size_hold_summary.json`, `data/size_hold_segments.csv`. The script is `size_hold.py`. There is no figure.

The controller's `height_source = size` mode holds ln(size) at a reference, with no OptiTrack, barometer or GPS in the loop. Results:
- **Coverage:** 26 hold segments of 5 s or more (368 s in total, 18 flights).
- **Height held by motion capture:** standard deviation **6.8 cm** median (7.1 cm time-weighted), and 27 cm peak to peak.
- **Vision vs. mocap during holds:** they agree only weakly (median R² 0.28), because the height only moves a few cm, which is the same order as the size noise.
- **Vision vs. mocap over a climb:** the size follows the height closely (R² 0.98), because the distance changes by tens of cm.

This is mostly a controller result. Put it in the paper only if the controller is part of it.

---

## 8. Flights and development history

Data comes from `data/tables/flights.csv` + `.tex` and `data/tables/versions.csv` + `.tex`. The script is `tables.py`.

### All climb attempts
| | |
|---|---|
| climb attempts (controller entered CLIMBING) | 28 |
| contact with the branch | 19 |
| **perched** (gripper closed and held) | **14** |
| aborted back to ALIGN | 6 |
| other aborts (TAG_LOST 1, WAIT_ARM 2) | 3 |

25 of the 28 climbs have a vision log; the 3 from the 10-12 morning don't. In all 25, the vision sent a point on at least 96.9% of frames during the climb, and on 98% or more in 24 of them. So **the aborts were not vision dropouts**: they are controller-side conditions. A wrong point can't be excluded this way, though.

Two notes:
- One "contact" came only 0.2 s into a climb (10-13 10:19, no size yet). It is probably a false contact detection; check before counting it.
- `contact_xy_spread_m` in the summary is how far apart the perch spots were across flights (different branches and places). It is not an accuracy figure.

### By code version
| Ver. | Change | Runs | Point % | Losses/min | Climbs | Contact | Perched |
|---|---|---|---|---|---|---|---|
| — | 10-12 morning, no vision log | 0 | – | – | 3 | 2 | 2 |
| V1 | stored target, point kept until reset | 4 | 74.0 | 7.18 | 7 | 2 | 1 |
| V2 | size self-heals, point held, features off crossings | 2 | 93.3 | 10.25 | 2 | 2 | 0 |
| V3 | hold through motion blur, no lock at image edge | 5 | 91.2 | 5.47 | 6 | 5 | 4 |
| V4 | lock only where size works, steadier size | 9 | 89.7 | 3.77 | 0 | 0 | 0 |
| V4+ | V4 + put-back checks in development | 14 | 88.2 | 5.63 | 9 | 7 | 6 |
| V5 | put-back window, locks off crossings | 1 | 87.2 | 3.11 | 1 | 1 | 1 |

From V3 onward, 11 of 16 climbs ended perched (69%) and 13 of 16 made contact, against 1 of 9 perched in V1–V2. V4 has no climbs: those flights tested the size height hold.

Most runs used the last commit plus uncommitted changes, which is the `*` in the labels. The version labels are as logged; they are not exact snapshots of the code. This table is a development history, **not a controlled ablation**: branches, light and the controller changed between days too.

---

## 9. Gaps before this is a paper

1. **Deployed-model accuracy.** You need mAP and precision/recall of the YOLOv8m-seg HEF on a held-out set; only the YOLO26 checkpoints' numbers are in the repo. Also say what the dataset is: images, how they were labelled, train/val split.
2. **Fly the prepared fixes (section 4) before quoting mask quality:** letterbox input, box offset, NMS format and the neighbour count. Otherwise report the masks as they are and call it a known limitation.
3. **Perch-point accuracy.** There is no ground truth for where on the branch the point should be. A small hand-labelled set would give a pixel error: about 100 frames, marking the intended point.
4. **Camera intrinsics are not calibrated** (DSJ camera). The s = K/d result doesn't need them. Anything in metres from the image alone does, for example lateral offset or branch diameter.
5. **A controlled comparison.** For example, KLT tracking vs. detector-only at 6 Hz on the same recorded flights. Running the detector-only pipeline on the 33 videos would give this without flying.
6. **Hailo at PCIe gen 3.** The HEF is multi-context, so it is bandwidth-bound. `dtparam=pciex1_gen=3` might raise the 13.5 FPS. Test on the ground first: the Hailo driver errors seen earlier happened at gen 2 already.

---

## 10. A possible story for the paper

- **Problem:** perch on thin branches with onboard compute only.
- **Approach:**
  - segmentation picks a vertical branch;
  - a cheap tracker carries the point at 27 Hz;
  - the branch's apparent width gives distance and time-to-contact without calibration or a depth sensor.
- **Speed:**
  - 165 ms per frame while searching, 37 ms while tracking;
  - 48 ms from camera to attitude command;
  - under 0.3 GB and about 1–3 W extra on a Pi 5 + Hailo-8.
- **Precision:**
  - width to about 0.2 px;
  - distance to 2.5%;
  - TTC to 7% in the 2–8 s range.
- **Robustness:**
  - a point on 88% of frames;
  - losses re-found in 0.46 s (37% at once);
  - losses driven by body rate, from 0.7% below 10°/s to 40% above 60°/s.
- **Outcome:** 14 perches in 28 climb attempts, 11 of 16 with the later versions (V3 onward).

---

## Figures

| File | Shows |
|---|---|
| `fig_frame_period` | frame period CDF, searching vs. tracking |
| `fig_latency` | latency CDFs, camera → UDP → attitude command |
| `fig_detection_time` | time per detection step |
| `fig_resources` | CPU, memory, power over idle → flight replay → detector only |
| `fig_detection_conf` | detection confidence by branch angle + angle histogram |
| `fig_mask_width` | mask width vs. edge width |
| `fig_decoder_issues` | mask pixels a correctly placed box would keep |
| `fig_input_modes` | candidate availability vs. confidence threshold, and mask width error, for four input preparations |
| `fig_tracking` | point/size availability per flight, re-find time CDF, put-back sizes (2 columns wide) |
| `fig_size_noise` | size noise vs. apparent width |
| `fig_loss_vs_rate` | probability of losing the target vs. body rate |
| `fig_size_distance` | apparent width vs. mocap distance, s = K/d |
| `fig_ttc` | TTC from size vs. true TTC |
| `fig_approach_example` | one climb: height and size over time |

## Re-running

Run from `paper/scripts/`, in this order:
1. **`export_bags.py`:** runs inside the `uav_ros_stack_binary` container; it writes `data/bags/`.
2. **`vision_runs.py`:** writes `data/vision/`.
3. **`fig_timing.py`, `fig_approach.py`:** timing and approach figures.
4. **`detect_survey.py`, `mask_vs_edge.py`, `fig_detection.py`, `input_modes.py`, `fig_input_modes.py`:** detection. The survey and the input test need the Hailo and stop if `main_record.py` is running.
5. **`resource_run.py`, `fig_resources.py`:** resources. Also needs the Hailo and has the same flight check.
6. **`fig_tracking.py`, `tables.py`, `size_hold.py`:** tracking, tables and the size hold.
