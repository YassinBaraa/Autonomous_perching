#!/usr/bin/env python3
"""Vision-side tables from the recordings: per-frame records (<run>_overlay.jsonl) and per-stage
timings (run_<run>.log). Writes paper/data/vision/: runs.csv (one row per run), <run>_frames.csv,
timing_detection.csv, timing_ibvs.csv, timing_main.csv."""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

REC = "/home/crow-black/Autonomous_perching/UDP_client/recordings"
OUT = "/home/crow-black/Autonomous_perching/paper/data/vision"
os.makedirs(OUT, exist_ok=True)

RE_DET = re.compile(r"\[DetectionPipeline\] Frame (\d+): TIMING — read=(\d+)ms detect=(\d+)ms track=(\d+)ms postprocess=(\d+)ms total=(\d+)ms")
RE_IB = re.compile(r"\[IBVSPipeline\] Frame (\d+): TIMING — source_read=(\d+)ms tracking_and_extraction=(\d+)ms controller=(\d+)ms frame_total=(\d+)ms")
RE_MAIN = re.compile(r"\[main\] Frame (\d+): TIMING — raw_write=(\d+)ms udp_send=(\d+)ms data_write=(\d+)ms loop_total=(\d+)ms since_prev_frame=(\d+)ms")


def read_jsonl(path):
    recs, header = [], None
    for line in open(path, errors="replace"):
        line = line.strip("\x00\n ")
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("header"):
            header = r
        else:
            recs.append(r)
    return header, recs


def main():
    runs = []
    det_rows, ib_rows, main_rows = [], [], []
    logs = {os.path.basename(p)[4:19]: p for p in glob.glob(f"{REC}/run_*.log")}
    jsonls = {os.path.basename(p)[:15]: p for p in glob.glob(f"{REC}/*_overlay.jsonl")}
    # a run's log and its jsonl differ by up to a second in their names
    def match_log(key):
        if key in logs:
            return logs[key]
        k = key[:13]
        cand = [v for kk, v in logs.items() if kk[:13] == k]
        return cand[0] if cand else None
    for key in sorted(jsonls):
        header, recs = read_jsonl(jsonls[key])
        if len(recs) < 20:
            continue
        logp = match_log(key)
        log = open(logp, errors="replace").read() if logp else ""
        mode = (header or {}).get("mode", "branch")
        t0 = recs[0]["t"]
        # per-frame table
        with open(f"{OUT}/{key}_frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t", "wall", "tf", "tstate", "ctrl", "px", "py", "size", "age", "point_x", "point_y",
                        "n_feat", "width", "stored_dx", "stored_dy", "views", "events"])
            for i, r in enumerate(recs, 1):
                p = r.get("packet") or {}
                pt = r.get("point") or [None, None]
                st = r.get("stored")
                br = r.get("branch") or {}
                sdx = sdy = None
                if st and r.get("point"):
                    sdx, sdy = st[0] - pt[0], st[1] - pt[1]
                w.writerow([i, f"{r['t'] - t0:.4f}", r.get("wall"), r.get("tf") and f"{r['tf'] - t0:.4f}", r.get("tstate"), r.get("ctrl"),
                            p.get("px"), p.get("py"), p.get("size"), p.get("age"), pt[0], pt[1], len(r.get("feat") or []),
                            br.get("width"), sdx, sdy, r.get("views"), "|".join(e[0] for e in r.get("events", []))])
        for m in RE_DET.finditer(log):
            det_rows.append([key] + [int(x) for x in m.groups()])
        for m in RE_IB.finditer(log):
            ib_rows.append([key] + [int(x) for x in m.groups()])
        for m in RE_MAIN.finditer(log):
            main_rows.append([key] + [int(x) for x in m.groups()])
        pk = [r["packet"] for r in recs if r.get("packet")]
        sizes = [p["size"] for p in pk if "size" in p]
        ages = [p["age"] for p in pk if "age" in p]
        ev = [e[0] for r in recs for e in r.get("events", [])]
        dur = recs[-1]["t"] - t0
        trk = [r for r in recs if r.get("tstate") == "tracking"]
        runs.append(dict(run=key, mode=mode, version=(header or {}).get("version"), log=bool(logp), frames=len(recs),
                         duration_s=round(dur, 2), fps=round(len(recs) / dur, 2) if dur > 0 else None,
                         packets=len(pk), point_pct=round(100 * len(pk) / len(recs), 1),
                         size_pct=round(100 * len(sizes) / max(len(pk), 1), 1),
                         width_min=round(min(sizes), 2) if sizes else None, width_max=round(max(sizes), 2) if sizes else None,
                         age_median_ms=round(1000 * float(np.median(ages)), 1) if ages else None,
                         tracking_pct=round(100 * len(trk) / len(recs), 1),
                         locks=ev.count("LOCKED"), losses=ev.count("LOST"), refinds=ev.count("RE-FOUND"),
                         putbacks=ev.count("PUT BACK"), resets=ev.count("RESET"), blur_skips=log.count("blurry frame skipped"),
                         held=any(r.get("ctrl") in ("CLIMBING", "STRIKE", "PUSH", "PERCHED") for r in recs)))
    with open(f"{OUT}/runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    for name, rows, cols in (("timing_detection", det_rows, ["run", "frame", "read", "detect", "track", "postprocess", "total"]),
                             ("timing_ibvs", ib_rows, ["run", "frame", "source_read", "tracking", "controller", "frame_total"]),
                             ("timing_main", main_rows, ["run", "frame", "raw_write", "udp_send", "data_write", "loop_total", "since_prev"])):
        with open(f"{OUT}/{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
    print(f"{len(runs)} runs; timing rows: detection {len(det_rows)}, ibvs {len(ib_rows)}, main {len(main_rows)}")


if __name__ == "__main__":
    main()
