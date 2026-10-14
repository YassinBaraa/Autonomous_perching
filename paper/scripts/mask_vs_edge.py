#!/usr/bin/env python3
"""Segmentation boundary accuracy: at every chosen perch point of the detection survey, the width
from the mask (2 x distance-to-boundary) vs the width measured on the image edges by the tracker's
BranchWidthTracker (half-contrast edges, sub-pixel, 17 cross-sections). CPU only.
Writes paper/data/detection/mask_vs_edge.csv."""
import csv
import io
import sys
from contextlib import redirect_stdout

import cv2
import numpy as np

sys.path.insert(0, "/home/crow-black/Autonomous_perching/ibvs")
from trackers.BranchWidthTracker import BranchWidthTracker, _image_direction  # noqa: E402

REC = "/home/crow-black/Autonomous_perching/UDP_client/recordings"
D = "/home/crow-black/Autonomous_perching/paper/data/detection"


def main():
    rows = list(csv.DictReader(open(f"{D}/best_points.csv")))
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    out = []
    for run, rs in by_run.items():
        want = {int(r["frame"]): r for r in rs}
        cap = cv2.VideoCapture(f"{REC}/{run}_raw.mp4")
        fi = 0
        while want:
            ok, f = cap.read()
            if not ok:
                break
            fi += 1
            r = want.pop(fi, None)
            if r is None:
                continue
            p = np.array([float(r["x"]), float(r["y"])])
            wm = 2.0 * float(r["half_px"])
            if wm <= 0:
                continue
            d = _image_direction(f, p, wm / 2)
            if d is None:
                continue
            bt = BranchWidthTracker()
            bt.expect(d, wm)
            with redirect_stdout(io.StringIO()):
                for _ in range(2):          # second call confirms (2 agreeing frames on the same image)
                    w, why, raw = bt.measure(f, p)
            if raw is None:
                continue
            out.append([run, fi, round(wm, 2), round(raw, 2), round(raw / wm, 4), r["angle"]])
    with open(f"{D}/mask_vs_edge.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["run", "frame", "mask_width_px", "edge_width_px", "edge_over_mask", "angle"])
        w.writerows(out)
    a = np.array([o[4] for o in out])
    print(f"{len(out)} points; edge/mask width: median {np.median(a):.3f}, IQR {np.percentile(a, 25):.3f}-{np.percentile(a, 75):.3f}")


if __name__ == "__main__":
    main()
