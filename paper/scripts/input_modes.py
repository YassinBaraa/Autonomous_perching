#!/usr/bin/env python3
"""How the 1280x720 camera frame is fed to the 640x640 network: four ways on the same recorded frames.
  squash     resize to 640x640 (as flown: 0.5 x, 0.89 y)
  letterbox  scale 0.5 to 640x360, grey bars above and below (how YOLOv8 is usually trained)
  crop640    the central 640x640 at full resolution (no scaling; sees 50 % of the width)
  crop720    the central 720x720 scaled to 640 (0.89 both ways; sees 56 % of the width)
All with the corrected decoder (+0.5-cell box offset, NMS on x,y,w,h). Compared inside the central
640x640 window that every mode sees: detections, confidence, and at the chosen perch point (current
post-processing, vertical-only rule) the mask width vs the width on the image edges (BranchWidthTracker).
Uses the Hailo; stops at once if main_record.py starts. Writes paper/data/detection/input_modes.csv."""
import csv
import glob
import io
import os
import sys
import time
from contextlib import redirect_stdout

import cv2
import numpy as np

DP = "/home/crow-black/Autonomous_perching/detection_pipeline"
REC = "/home/crow-black/Autonomous_perching/UDP_client/recordings"
OUT = "/home/crow-black/Autonomous_perching/paper/data/detection"
sys.path.insert(0, DP)
from detectors.HailoSegDetector import HailoSegDetector, _dfl_decode, _sigmoid, INPUT_SIZE  # noqa: E402
from pipeline.PipelineContext import PipelineContext  # noqa: E402
from postprocessing.masks.MaskExtraction import MaskExtraction  # noqa: E402
from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap  # noqa: E402
from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton  # noqa: E402
from postprocessing.scoring import CandidateScoring as CS  # noqa: E402
sys.path.insert(0, "/home/crow-black/Autonomous_perching/ibvs")
from trackers.BranchWidthTracker import BranchWidthTracker, _image_direction  # noqa: E402
from detect_survey import flight_running  # noqa: E402

W, H = 1280, 720
WIN = (320, 40, 960, 680)          # x0, y0, x1, y1: the window every mode sees
PAD = 114                          # Ultralytics letterbox grey


def prep(rgb, mode):
    """(640x640 input, (sx, ox, sy, oy)) with frame = input * s + o."""
    if mode == "squash":
        return cv2.resize(rgb, (640, 640)), (W / 640, 0.0, H / 640, 0.0)
    if mode == "letterbox":
        inp = np.full((640, 640, 3), PAD, np.uint8)
        inp[140:500] = cv2.resize(rgb, (640, 360))
        return inp, (2.0, 0.0, 2.0, -280.0)
    if mode == "crop640":
        return np.ascontiguousarray(rgb[40:680, 320:960]), (1.0, 320.0, 1.0, 40.0)
    if mode == "crop720":
        return cv2.resize(rgb[:, 280:1000], (640, 640)), (720 / 640, 280.0, 720 / 640, 0.0)
    raise ValueError(mode)


def decode(fo, m, conf):
    """Corrected decoder: boxes from the cell centre, NMS on x,y,w,h, masks mapped back through m."""
    sx, ox, sy, oy = m
    proto, box_map, cls_map, coeff_map = None, {}, {}, {}
    for arr in fo.values():
        a = arr.squeeze(0) if arr.ndim == 4 and arr.shape[0] == 1 else arr
        h, w, c = a.shape
        if h == 160 and w == 160 and c == 32:
            proto = a.transpose(2, 0, 1)
        elif c == 64:
            box_map[INPUT_SIZE // h] = a
        elif c == 1:
            cls_map[INPUT_SIZE // h] = a[..., 0]
        elif c == 32:
            coeff_map[INPUT_SIZE // h] = a
    cells = []
    for stride in (8, 16, 32):
        sc = cls_map[stride]
        ltrb = _dfl_decode(box_map[stride])
        for iy, ix in np.argwhere(sc > conf):
            l, t, r, b = ltrb[iy, ix]
            box = [(ix + 0.5 - l) * stride * sx + ox, (iy + 0.5 - t) * stride * sy + oy,
                   (ix + 0.5 + r) * stride * sx + ox, (iy + 0.5 + b) * stride * sy + oy]
            box = [float(np.clip(box[0], 0, W)), float(np.clip(box[1], 0, H)), float(np.clip(box[2], 0, W)), float(np.clip(box[3], 0, H))]
            cells.append((box, float(sc[iy, ix]), coeff_map[stride][iy, ix]))
    if not cells:
        return []
    idx = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b, _, _ in cells], [s for _, s, _ in cells], conf, 0.45)
    dets = []
    for i in np.array(idx).flatten():
        box, score, coef = cells[i]
        logit = np.einsum("c,chw->hw", coef, proto).astype(np.float32)
        # proto pixel u covers input pixels 4u..4u+3 (centre 4u + 2 in continuous coordinates); frame pixel
        # index = continuous coordinate - 0.5
        A = np.float32([[4 * sx, 0, 2 * sx + ox - 0.5], [0, 4 * sy, 2 * sy + oy - 0.5]])
        full = cv2.warpAffine(logit, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-20.0) > 0.0
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        mk = np.zeros((H, W), bool)
        mk[y1:y2, x1:x2] = full[y1:y2, x1:x2]
        dets.append((box, score, mk))
    return dets


def in_win(x, y):
    return WIN[0] <= x < WIN[2] and WIN[1] <= y < WIN[3]


def main():
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    modes = ("squash", "letterbox", "crop640", "crop720")
    det = HailoSegDetector(hef_path=f"{DP}/model/yolov8_segmentation.hef", conf=0.5)
    stages = [MaskExtraction(), DistanceHeatmap(), BitmaskSkeleton()]
    CS.VERTICAL_ONLY = True
    f = open(f"{OUT}/input_modes.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["run", "frame", "mode", "t_prep_ms", "n_det_win", "max_conf_win", "best_x", "best_y", "best_conf", "best_angle",
                "mask_width_px", "edge_width_px"])
    videos = sorted(p for p in glob.glob(f"{REC}/2026101[23]_*_raw.mp4") if os.path.exists(p.replace("_raw.mp4", "_overlay.jsonl")))
    n, t_start = 0, time.time()
    try:
        for vid in videos:
            run = os.path.basename(vid)[:15]
            cap = cv2.VideoCapture(vid)
            fi = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fi += 1
                if fi % every:
                    continue
                if n % 5 == 0 and flight_running():
                    print("vision started -- stopping", flush=True)
                    return
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for mode in modes:
                    a = time.perf_counter()
                    inp, m = prep(rgb, mode)
                    tp = 1000 * (time.perf_counter() - a)
                    det._bindings.input().set_buffer(inp)
                    det._configured.run([det._bindings], timeout=1000)
                    fo = {k: (v.astype(np.float32) - det._quant[k][1]) * det._quant[k][0] for k, v in det._out_bufs.items()}
                    dets = decode(fo, m, det.conf)
                    win = [d for d in dets if in_win((d[0][0] + d[0][2]) / 2, (d[0][1] + d[0][3]) / 2)]
                    row = [run, fi, mode, round(tp, 2), len(win), round(max((d[1] for d in win), default=0.0), 4)]
                    best = None
                    if dets:
                        ctx = PipelineContext(frame=frame)
                        ctx.masks = np.array([d[2] for d in dets], bool)
                        ctx.tracks = [(tuple(d[0]), d[2], d[1], 0, i + 1, {}) for i, d in enumerate(dets)]
                        with redirect_stdout(io.StringIO()):
                            for st in stages:
                                ctx = st.run(ctx)
                            ctx = CS.CandidateScoring().run(ctx)
                        best = ctx.best_candidate
                    if best is not None and in_win(best["x"], best["y"]):
                        p = np.array([float(best["x"]), float(best["y"])])
                        wm = 2.0 * float(best.get("distance_to_boundary") or 0.0)
                        raw = None
                        d = _image_direction(frame, p, wm / 2) if wm > 0 else None
                        if d is not None:
                            bt = BranchWidthTracker()
                            bt.expect(d, wm)
                            with redirect_stdout(io.StringIO()):
                                for _ in range(2):
                                    _, _, raw = bt.measure(frame, p)
                        row += [best["x"], best["y"], round(best["confidence"], 4), round(best.get("angle_deg") or -1, 1),
                                round(wm, 2), None if raw is None else round(raw, 2)]
                    else:
                        row += [None] * 6
                    w.writerow(row)
                n += 1
                if n % 50 == 0:
                    f.flush()
                    print(f"{n} frames, {time.time() - t_start:.0f} s", flush=True)
    finally:
        det.release()
        f.close()
        print(f"done: {n} frames in {time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
