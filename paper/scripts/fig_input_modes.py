#!/usr/bin/env python3
"""Input preparation for the network (input_modes.py): share of frames with a perch point in the central
640x640 window vs the confidence threshold, and mask width error vs the image edges, per mode.
Figure fig_input_modes; data input_modes_summary.json."""
import numpy as np

import common as c
from common import plt

MODES = [("squash", "squash (as flown)", c.C["red"], "o-"), ("letterbox", "letterbox", c.C["blue"], "s-"),
         ("crop720", "crop 720² → 640²", c.C["green"], "^--"), ("crop640", "crop 640² (full res.)", c.C["orange"], "d--")]


def main():
    d = c.load_csv(f"{c.DATA}/detection/input_modes.csv")
    th = np.round(np.arange(0.5, 0.951, 0.05), 2)
    summ = {}
    fig, axs = plt.subplots(1, 2, figsize=(c.COL2 * 0.62, 1.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for m, lab, col, mk in MODES:
        s = d["mode"] == m
        n = int(s.sum())
        b = d["best_conf"][s]
        ok = np.isfinite(b)
        avail = [100.0 * np.sum(b[ok] >= t) / n for t in th]
        axs[0].plot(th, avail, mk, color=col, ms=2.5, label=lab)
        ew, mw = d["edge_width_px"][s], d["mask_width_px"][s]
        g = np.isfinite(ew) & np.isfinite(mw) & (mw > 0)
        err = np.abs(ew[g] - mw[g])
        summ[m] = dict(frames=n, point_pct_by_threshold={f"{t:.2f}": round(a, 1) for t, a in zip(th, avail)},
                       detection_in_window_pct=round(100 * float(np.mean(d["n_det_win"][s] > 0)), 1),
                       max_conf_median=c.pct(d["max_conf_win"][s][d["n_det_win"][s] > 0], 50),
                       mask_width_abs_err_px_median=c.pct(err, 50), edge_over_mask_median=c.pct(ew[g] / mw[g], 50),
                       prep_ms_median=c.pct(d["t_prep_ms"][s], 50))
    axs[0].set_xlabel("confidence threshold")
    axs[0].set_ylabel("frames with a perch point [%]")
    axs[0].legend(loc="lower left", fontsize=6)
    axs[0].set_ylim(40, 100)
    names = [lab.split(" (")[0] for _, lab, _, _ in MODES]
    errs = [summ[m]["mask_width_abs_err_px_median"] for m, _, _, _ in MODES]
    axs[1].bar(range(len(MODES)), errs, color=[col for _, _, col, _ in MODES])
    axs[1].set_xticks(range(len(MODES)))
    axs[1].set_xticklabels(["squash", "letterbox", "crop 720", "crop 640"], rotation=30, ha="right", fontsize=6.5)
    axs[1].set_ylabel("mask width error [px]")
    axs[1].grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    c.save(fig, "fig_input_modes")
    summ["note"] = ("all modes with the corrected decoder; compared inside the central 640x640 window that every mode sees; "
                    "mask width error = |mask width - edge width| at the chosen point (median)")
    c.write_json("input_modes_summary", summ)
    for m in summ:
        print(m, summ[m])


if __name__ == "__main__":
    main()
