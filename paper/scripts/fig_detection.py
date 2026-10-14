#!/usr/bin/env python3
"""Detector statistics from the survey (detect_survey.py): where the time of a detection frame goes,
what is detected (confidence by orientation), mask width vs true edge width, and the two decoder
issues (box half-cell offset clipping the masks, NMS box format).
Figures fig_detection_time, fig_detection_conf, fig_mask_width, fig_decoder_issues; data detection_summary.json."""
import numpy as np

import common as c
from common import plt

D = f"{c.DATA}/detection"


def main():
    fr = c.load_csv(f"{D}/frames.csv")
    de = c.load_csv(f"{D}/detections.csv")
    mv = c.load_csv(f"{D}/mask_vs_edge.csv")
    hb = c.load_csv  # noqa

    has = fr["n_det"] > 0
    steps = [("t_prep", "colour + resize to 640²"), ("t_hailo", "Hailo-8 inference"), ("t_dequant", "de-quantise outputs"),
             ("t_decode", "decode boxes + NMS"), ("t_masks", "assemble masks"), ("t_mask_merge", "merge masks"),
             ("t_heat", "distance transform"), ("t_skeleton", "skeletonise"), ("t_scoring", "choose point"),
             ("t_visualizer", "debug image")]
    med = {k: float(np.median(fr[k][has])) for k, _ in steps}
    p90 = {k: float(np.percentile(fr[k][has], 90)) for k, _ in steps}
    total = sum(med.values())

    ang = de["angle"]
    vert, hori = ang >= 60, ang <= 20
    diag = ~vert & ~hori & np.isfinite(ang)
    conf = de["conf"]
    clip_lost = (de["unclipped_px"] - de["mask_px"]) / np.maximum(de["unclipped_px"], 1)
    fixed_gain = (de["mask_px_fixed_box"] - de["mask_px"]) / np.maximum(de["mask_px"], 1)
    nms_more = fr["n_det_xywh"] > fr["n_det"]
    old_non_vertical = np.isfinite(fr["best_old_angle"]) & (fr["best_old_angle"] < 60)
    ratio = mv["edge_over_mask"]

    summ = dict(
        frames=int(len(fr["n_det"])), frames_with_detection=int(has.sum()), detections=int(len(conf)),
        detections_per_frame=dict(mean=float(np.mean(fr["n_det"])), max=int(np.max(fr["n_det"]))),
        detection_frame_ms_median=med, detection_frame_ms_p90=p90, detection_frame_ms_sum_of_medians=total,
        confidence=dict(vertical=dict(n=int(vert.sum()), median=float(np.median(conf[vert]))),
                        horizontal=dict(n=int(hori.sum()), median=float(np.median(conf[hori])) if hori.any() else None),
                        diagonal=dict(n=int(diag.sum()), median=float(np.median(conf[diag])) if diag.any() else None)),
        mask_width_vs_edges=dict(n=int(len(ratio)), edge_over_mask_median=float(np.median(ratio)),
                                 iqr=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
                                 mask_overestimate_factor=float(1 / np.median(ratio))),
        box_offset=dict(corrected_box_mask_gain_median=float(np.nanmedian(fixed_gain)),
                        corrected_box_mask_gain_p90=float(np.nanpercentile(fixed_gain, 90)),
                        detections_gaining_over_10pct=float(np.nanmean(fixed_gain > 0.10))),
        nms_format=dict(frames_where_correct_format_keeps_more=float(np.mean(nms_more[has])),
                        extra_detections_total=int(np.sum(fr["n_det_xywh"] - fr["n_det"]))),
        old_rule_picked_non_vertical=float(np.mean(old_non_vertical[np.isfinite(fr["best_old_angle"])])),
        new_rule_frames_with_candidate=float(np.mean(fr["n_cand_new"][has] > 0)),
        old_rule_frames_with_candidate=float(np.mean(fr["n_cand_old"][has] > 0)))
    c.write_json("detection_summary", summ)

    # --- where a detection frame's time goes (horizontal stacked bar + p90 whiskers per step)
    fig, ax = plt.subplots(figsize=(c.COL1, 2.0))
    names = [lab for _, lab in steps][::-1]
    vals = [med[k] for k, _ in steps][::-1]
    errs = [p90[k] - med[k] for k, _ in steps][::-1]
    colors = [c.C["orange"] if k == "t_hailo" else (c.C["blue"] if k in ("t_prep", "t_dequant", "t_decode", "t_masks") else c.C["green"])
              for k, _ in steps][::-1]
    ax.barh(names, vals, xerr=[np.zeros(len(errs)), errs], color=colors, height=0.6, error_kw=dict(lw=0.6, capsize=1.5))
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + e + 1.2, i, f"{v:.1f}", va="center", fontsize=6.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c.C["orange"], label="accelerator"), Patch(color=c.C["blue"], label="host: network I/O"),
                       Patch(color=c.C["green"], label="host: post-processing")], loc="center right", bbox_to_anchor=(1.0, 0.6), fontsize=6)
    ax.set_xlabel("time per detection frame [ms] (median, whisker = 90th pct.)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.12)
    c.save(fig, "fig_detection_time")

    # --- confidence by orientation + orientation histogram
    fig, axs = plt.subplots(1, 2, figsize=(c.COL1, 1.7), gridspec_kw=dict(width_ratios=[1.2, 1]))
    bins = np.linspace(0.5, 1.0, 21)
    for mask, col, lab in ((vert, c.C["green"], "vertical (≥60°)"), (diag, c.C["grey"], "diagonal"), (hori, c.C["red"], "horizontal (≤20°)")):
        if mask.any():
            axs[0].hist(conf[mask], bins=bins, color=col, alpha=0.65, label=f"{lab}, n={mask.sum()}")
    axs[0].set_xlabel("detection confidence")
    axs[0].set_ylabel("detections")
    axs[0].legend(loc="upper left", fontsize=6)
    axs[1].hist(ang[np.isfinite(ang)], bins=np.arange(0, 91, 5), color=c.C["blue"])
    axs[1].axvline(60, color=c.C["red"], lw=0.8, ls="--")
    axs[1].set_xlabel("branch angle [°]")
    axs[1].set_xticks([0, 30, 60, 90])
    fig.tight_layout(w_pad=0.6)
    c.save(fig, "fig_detection_conf")

    # --- mask width vs edge width
    fig, ax = plt.subplots(figsize=(c.COL1, 2.0))
    ax.scatter(mv["mask_width_px"], mv["edge_width_px"], s=2, color=c.C["blue"], alpha=0.4, lw=0)
    lim = [0, max(np.percentile(mv["mask_width_px"], 99), 10)]
    ax.plot(lim, lim, "k--", lw=0.7, label="edge = mask")
    mw, ew = mv["mask_width_px"], mv["edge_width_px"]
    centers, meds = [], []
    for v in np.unique(np.round(mw)):
        sel = np.round(mw) == v
        if sel.sum() >= 5:
            centers.append(v); meds.append(np.median(ew[sel]))
    ax.plot(centers, meds, "o-", color=c.C["red"], ms=2.5, lw=0.9, label="median edge width per mask width")
    summ["mask_width_vs_edges"]["binned"] = [[float(a), float(b)] for a, b in zip(centers, meds)]
    c.write_json("detection_summary", summ)
    ax.set_xlim(lim); ax.set_ylim(0, lim[1])
    ax.set_xlabel("width from segmentation mask [px]")
    ax.set_ylabel("width from image edges [px]")
    ax.legend(loc="upper left")
    c.save(fig, "fig_mask_width")

    # --- decoder issue: mask pixels a correctly placed box (+0.5 cell) would keep
    fig, ax = plt.subplots(figsize=(c.COL1, 1.7))
    g = 100 * fixed_gain[np.isfinite(fixed_gain)]
    ax.hist(np.clip(g, -20, 100), bins=np.linspace(-20, 100, 49), color=c.C["orange"])
    ax.axvline(np.median(g), color="k", lw=0.8, ls="--")
    ax.text(np.median(g) + 2, ax.get_ylim()[1] * 0.85, f"median +{np.median(g):.0f} %", fontsize=6.5)
    ax.set_xlabel("mask pixels kept with the corrected box [% of current]")
    ax.set_ylabel("detections")
    c.save(fig, "fig_decoder_issues")
    print({k: round(v, 1) for k, v in med.items()}, "sum", round(total, 1))
    print("conf", summ["confidence"], "\nmask", summ["mask_width_vs_edges"], "\nbox", summ["box_offset"], "\nnms", summ["nms_format"])
    print("old rule non-vertical picks", summ["old_rule_picked_non_vertical"], "| candidate available old/new",
          summ["old_rule_frames_with_candidate"], summ["new_rule_frames_with_candidate"])


if __name__ == "__main__":
    main()
