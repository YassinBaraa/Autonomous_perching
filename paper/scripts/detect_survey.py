#!/usr/bin/env python3
"""Detection survey: the deployed detector and post-processing on sampled frames of every branch
recording (2026-10-12/13), timing each sub-step and recording what was detected. Also measures two
decoder issues found while reading the code: NMS given (x1,y1,x2,y2) where it expects (x,y,w,h),
and boxes missing YOLOv8's +0.5-cell anchor offset (masks are clipped to those boxes).
Uses the Hailo; stops at once if main_record.py starts.
Writes paper/data/detection/frames.csv, detections.csv, best_points.csv.
usage: python3 detect_survey.py [every_n_frames]"""
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

DP = "/home/crow-black/Autonomous_perching/detection_pipeline"
REC = "/home/crow-black/Autonomous_perching/UDP_client/recordings"
OUT = "/home/crow-black/Autonomous_perching/paper/data/detection"
sys.path.insert(0, DP)
from detectors.HailoSegDetector import HailoSegDetector, _sigmoid, _dfl_decode, INPUT_SIZE   # noqa: E402
from pipeline.PipelineContext import PipelineContext                # noqa: E402
from postprocessing.masks.MaskExtraction import MaskExtraction       # noqa: E402
from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap  # noqa: E402
from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton  # noqa: E402
from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer  # noqa: E402
import postprocessing.scoring.CandidateScoring as CS                 # noqa: E402


def flight_running():
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if argv and b"python" in argv[0].split(b"/")[-1] and any(a.endswith(b"main_record.py") for a in argv[1:3]):
            return True
    return False


def raw_cells(fo, W, H, conf):
    """Every grid cell above `conf`, decoded exactly like HailoSegDetector._parse_seg_outputs
    (boxes as coded) plus the same box with YOLOv8's +0.5-cell anchor offset."""
    proto, box_map, cls_map, coeff_map = None, {}, {}, {}
    for arr in fo.values():
        a = arr.squeeze(0) if arr.ndim == 4 and arr.shape[0] == 1 else arr
        if a.ndim != 3:
            continue
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
    sx, sy = W / INPUT_SIZE, H / INPUT_SIZE
    for stride in (8, 16, 32):
        if stride not in box_map or stride not in cls_map:
            continue
        sc = cls_map[stride]
        ltrb = _dfl_decode(box_map[stride])
        for iy, ix in np.argwhere(sc > conf):
            l, t, r, b = ltrb[iy, ix]
            coded = [(ix - l) * stride * sx, (iy - t) * stride * sy, (ix + r) * stride * sx, (iy + b) * stride * sy]
            fixed = [(ix + 0.5 - l) * stride * sx, (iy + 0.5 - t) * stride * sy, (ix + 0.5 + r) * stride * sx, (iy + 0.5 + b) * stride * sy]
            clip = lambda v, m: float(np.clip(v, 0, m))
            cells.append(dict(stride=stride, score=float(sc[iy, ix]), coef=coeff_map[stride][iy, ix],
                              coded=[clip(coded[0], W), clip(coded[1], H), clip(coded[2], W), clip(coded[3], H)],
                              fixed=[clip(fixed[0], W), clip(fixed[1], H), clip(fixed[2], W), clip(fixed[3], H)]))
    return cells, proto


def nms(cells, key, xywh, conf):
    if not cells:
        return []
    boxes = [c[key] if not xywh else [c[key][0], c[key][1], c[key][2] - c[key][0], c[key][3] - c[key][1]] for c in cells]
    idx = cv2.dnn.NMSBoxes([list(map(float, b)) for b in boxes], [c["score"] for c in cells], conf, 0.45)
    return [cells[i] for i in np.array(idx).flatten()] if len(idx) else []


def main():
    every = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    os.makedirs(OUT, exist_ok=True)
    det = HailoSegDetector(hef_path=f"{DP}/model/yolov8_segmentation.hef", conf=0.5)
    stages = [("mask", MaskExtraction()), ("heat", DistanceHeatmap()), ("skeleton", BitmaskSkeleton())]
    vis = CandidateVisualizer()
    fcsv, dcsv, bcsv = (open(f"{OUT}/{n}.csv", "w", newline="") for n in ("frames", "detections", "best_points"))
    fw, dw, bw = csv.writer(fcsv), csv.writer(dcsv), csv.writer(bcsv)
    fw.writerow(["run", "frame", "t_prep", "t_hailo", "t_dequant", "t_decode", "t_masks", "t_mask_merge", "t_heat", "t_skeleton",
                 "t_scoring", "t_visualizer", "n_cells", "n_det", "n_det_xywh", "heat_max_px",
                 "best_old_angle", "best_old_conf", "best_old_x", "best_old_y", "best_new_angle", "best_new_conf", "best_new_x", "best_new_y",
                 "n_cand_old", "n_cand_new"])
    dw.writerow(["run", "frame", "det", "conf", "stride", "x1", "y1", "x2", "y2", "angle", "mask_px", "unclipped_px",
                 "mask_px_fixed_box", "median_half_px", "width_frac", "height_frac"])
    bw.writerow(["run", "frame", "x", "y", "half_px", "angle", "conf"])
    videos = sorted(p for p in glob.glob(f"{REC}/2026101[23]_*_raw.mp4") if os.path.exists(p.replace("_raw.mp4", "_overlay.jsonl")))
    n_frames, t_start = 0, time.time()
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
                if n_frames % 10 == 0 and flight_running():
                    print("vision started -- stopping the survey", flush=True)
                    return
                H, W = frame.shape[:2]
                t0 = time.perf_counter()
                rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (det._w, det._h))
                t1 = time.perf_counter()
                det._bindings.input().set_buffer(rgb)
                det._configured.run([det._bindings], timeout=1000)
                t2 = time.perf_counter()
                fo = {k: (v.astype(np.float32) - det._quant[k][1]) * det._quant[k][0] for k, v in det._out_bufs.items()}
                t3 = time.perf_counter()
                dets, proto = det._parse_seg_outputs(fo, W, H)          # the deployed decoder, as coded
                t4 = time.perf_counter()
                masks = []
                for d in dets:                                           # the deployed mask code, as coded
                    m = cv2.resize(_sigmoid(np.einsum('c,chw->hw', d[5], proto)), (W, H)) > 0.5
                    c = np.zeros((H, W), bool)
                    c[d[1]:d[3], d[0]:d[2]] = m[d[1]:d[3], d[0]:d[2]]
                    masks.append(c)
                t5 = time.perf_counter()
                # the two issues, measured on the same raw cells
                cells, _ = raw_cells(fo, W, H, det.conf)
                kept_coded = nms(cells, "coded", False, det.conf)
                kept_xywh = nms(cells, "coded", True, det.conf)
                ctx = PipelineContext(frame=frame)
                ctx.masks = np.array(masks, bool) if masks else None
                tt = {}
                for name, st in stages:
                    a = time.perf_counter(); ctx = st.run(ctx); tt[name] = time.perf_counter() - a
                hm = ctx.debug.get("heatmap_max_px", 0.0)
                tracks = [((d[0], d[1], d[2], d[3]), masks[i], d[4], 0, i + 1, {}) for i, d in enumerate(dets)]
                res = {}
                for label, vertical in (("old", False), ("new", True)):
                    CS.VERTICAL_ONLY = vertical
                    c2 = PipelineContext(frame=frame)
                    c2.tracks, c2.binary_mask, c2.distance_heatmap, c2.skeleton = tracks, ctx.binary_mask, ctx.distance_heatmap, ctx.skeleton
                    c2.debug["heatmap_max_px"] = hm
                    a = time.perf_counter(); c2 = CS.CandidateScoring().run(c2); tt[f"scoring_{label}"] = time.perf_counter() - a
                    res[label] = c2
                CS.VERTICAL_ONLY = True
                a = time.perf_counter(); vis.run(res["new"]); tt["vis"] = time.perf_counter() - a
                bo, bn = res["old"].best_candidate, res["new"].best_candidate
                g = lambda b, k: None if b is None or b.get(k) is None else (round(b[k], 3) if isinstance(b[k], float) else b[k])
                fw.writerow([run, fi, *(round(1000 * v, 2) for v in (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)),
                             *(round(1000 * tt[k], 2) for k in ("mask", "heat", "skeleton", "scoring_new", "vis")),
                             len(cells), len(kept_coded), len(kept_xywh), round(hm, 2),
                             g(bo, "angle_deg"), g(bo, "confidence"), g(bo, "x"), g(bo, "y"),
                             g(bn, "angle_deg"), g(bn, "confidence"), g(bn, "x"), g(bn, "y"),
                             len(res["old"].candidates or []), len(res["new"].candidates or [])])
                # per detection, matched to its raw cell by score+box (the same cell NMS kept)
                for i, d in enumerate(dets):
                    cell = min(kept_coded, key=lambda c: abs(c["score"] - d[4]) + sum(abs(a - b) for a, b in zip(c["coded"], d[:4]))) if kept_coded else None
                    m = masks[i]
                    full = cv2.resize(_sigmoid(np.einsum('c,chw->hw', d[5], proto)), (W, H)) > 0.5
                    fixed_px = None
                    if cell is not None:
                        x1, y1, x2, y2 = (int(v) for v in cell["fixed"])
                        fixed_px = int(full[y1:y2, x1:x2].sum())
                    ang = CS._branch_angle(m, ctx.skeleton, (d[0], d[1], d[2], d[3]), W, H)
                    sk = (ctx.skeleton > 0) & m if ctx.skeleton is not None else None
                    half = float(np.median(ctx.distance_heatmap[sk] * hm)) if sk is not None and sk.any() else None
                    dw.writerow([run, fi, i + 1, round(d[4], 4), cell and cell["stride"], d[0], d[1], d[2], d[3],
                                 None if ang is None else round(ang, 1), int(m.sum()), int(full.sum()), fixed_px,
                                 None if half is None else round(half, 2), round((d[2] - d[0]) / W, 3), round((d[3] - d[1]) / H, 3)])
                if bn:
                    bw.writerow([run, fi, bn["x"], bn["y"], round(bn.get("distance_to_boundary") or 0.0, 2), g(bn, "angle_deg"), g(bn, "confidence")])
                n_frames += 1
                if n_frames % 100 == 0:
                    print(f"{n_frames} frames, {time.time() - t_start:.0f} s", flush=True)
                    for f in (fcsv, dcsv, bcsv):
                        f.flush()
    finally:
        det.release()
        for f in (fcsv, dcsv, bcsv):
            f.close()
        print(f"done: {n_frames} frames from {len(videos)} videos in {time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
