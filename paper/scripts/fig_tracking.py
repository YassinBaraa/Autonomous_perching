#!/usr/bin/env python3
"""Tracking quality from the flight recordings (35 runs) and the flight bags:
continuity (share of frames with a point), losses per minute of tracking, re-find latency,
put-back sizes (log), size availability and size noise (detrended) by width, and what drives
losses: UAV body rate (IMU in the bags) in 0.5 s windows with and without a loss.
Figures fig_tracking, fig_size_noise, fig_loss_vs_rate; data tracking_summary.json, runs_metrics.csv."""
import csv
import glob
import re

import numpy as np

import common as c
from common import plt

RE_PB = re.compile(r"Frame (\d+): point (\d+)px off the stored target \((\d+) matches\)")
HALF = 4
FAR_PX = 15.0


def _detrended(size, frame):
    """(size, |residual| of log size from a line fit over frames i-HALF..i+HALF) for every frame whose
    whole window is consecutive and has a size."""
    out = []
    x = np.arange(-HALF, HALF + 1, dtype=float)
    for i in range(HALF, len(size) - HALF):
        w = size[i - HALF:i + HALF + 1]
        if not np.all(np.isfinite(w)) or frame[i + HALF] - frame[i - HALF] != 2 * HALF:
            continue
        y = np.log(w)
        a, b = np.polyfit(x, y, 1)
        out.append((size[i], abs(y[HALF] - b)))
    return np.array(out).reshape(-1, 2)


def main():
    r = c.runs()
    pairs = c.pair_runs_bags()
    refind, putback, size_steps, losses_t, widths_t = [], [], [], {}, {}
    per_run = []
    for run in r["run"]:
        fr = c.frames(run)
        t, ev, st = fr["t"], fr["events"], fr["tstate"]
        trk_min = float(np.sum(np.diff(t)[st[:-1] == "tracking"])) / 60.0
        lost_t = [t[i] for i in range(len(t)) if "LOST" in str(ev[i]).split("|")]
        found_t = [t[i] for i in range(len(t)) if "RE-FOUND" in str(ev[i]).split("|")]
        for lt in lost_t:
            nxt = [ft for ft in found_t if ft >= lt]
            if nxt and nxt[0] - lt < 60:
                refind.append(nxt[0] - lt)
        losses_t[run] = (np.array([fr["wall"][i] for i in range(len(t)) if "LOST" in str(ev[i]).split("|")]),
                         fr["wall"][st == "tracking"])
        # size noise: residual of log(size) from a straight line over 9 consecutive frames (removes the
        # true change while approaching), at the middle frame
        size_steps.append(_detrended(fr["size"], fr["frame"]))
        widths_t[run] = (fr["wall"], fr["size"])
        per_run.append(dict(run=run, tracking_min=round(trk_min, 3), point_pct=float(r["point_pct"][r["run"] == run][0]),
                            size_pct=float(r["size_pct"][r["run"] == run][0]), losses=len(lost_t),
                            losses_per_min=round(len(lost_t) / trk_min, 2) if trk_min > 0 else None,
                            putbacks=int(np.sum([("PUT BACK" in str(e)) for e in ev])),
                            version=str(r["version"][r["run"] == run][0]).split("·")[-1].strip()))
    # put-back magnitudes from the logs
    for p in glob.glob(f"{c.REC}/run_2026101[23]_*.log"):
        for m in RE_PB.finditer(open(p, errors="replace").read()):
            putback.append((int(m.group(2)), int(m.group(3))))
    putback = np.array(putback, float)
    refind = np.array(refind)
    S = np.vstack(size_steps)

    # losses vs angular rate (IMU in the bag, same clock as the vision's wall time)
    win_rates, loss_rates, win_far = [], [], []
    for run, b in pairs.items():
        imu = c.bag(b, "mavros/imu/data")
        if imu is None or not len(imu["t"]):
            continue
        rate = np.degrees(np.sqrt(imu["wx"] ** 2 + imu["wy"] ** 2 + imu["wz"] ** 2))
        lt, trk_wall = losses_t[run]
        ww, ws = widths_t[run]
        if not len(trk_wall):
            continue
        # 0.5 s windows over tracked time: max roll/pitch rate in the window, and whether a loss happened in it
        edges = np.arange(trk_wall[0], trk_wall[-1], 0.5)
        for a in edges:
            if not np.any((trk_wall >= a) & (trk_wall < a + 0.5)):
                continue
            m = (imu["t"] >= a - 0.25) & (imu["t"] < a + 0.5)
            if m.sum() < 3:
                continue
            win_rates.append(float(np.max(rate[m])))
            loss_rates.append(float(np.any((lt >= a) & (lt < a + 0.5))))
            # branch width just before the window (last size sent in the second before it)
            k = (ww >= a - 1.0) & (ww < a) & np.isfinite(ws)
            win_far.append(bool(k.any() and ws[k][-1] < FAR_PX))
    win_rates, loss_rates, win_far = np.array(win_rates), np.array(loss_rates), np.array(win_far)
    bins = np.array([0, 10, 20, 30, 45, 60, 90, 360])

    def loss_curve(sel):
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = sel & (win_rates >= lo) & (win_rates < hi)
            out.append(dict(lo=int(lo), hi=int(hi), n=int(m.sum()), rate_median=c.pct(win_rates[m], 50) if m.any() else None,
                            p_loss=float(loss_rates[m].mean()) if m.sum() >= 20 else None))
        return out
    p_loss = loss_curve(np.ones(len(win_rates), bool))
    p_loss_far = loss_curve(win_far)

    wb = [(0, 8), (8, 15), (15, 40), (40, 200)]
    noise = {}
    for lo, hi in wb:
        m = (S[:, 0] >= lo) & (S[:, 0] < hi)
        noise[f"{lo}-{hi} px"] = dict(n=int(m.sum()), median_pct=100 * c.pct(S[m, 1], 50), p90_pct=100 * c.pct(S[m, 1], 90),
                                      median_px=c.pct(S[m, 0] * S[m, 1], 50))
    noise["all"] = dict(n=int(len(S)), median_pct=100 * c.pct(S[:, 1], 50), p90_pct=100 * c.pct(S[:, 1], 90),
                        median_px=c.pct(S[:, 0] * S[:, 1], 50))
    tot_min = sum(x["tracking_min"] for x in per_run)
    summ = dict(runs=len(per_run), tracking_minutes=round(tot_min, 1),
                point_pct_median=float(np.median([x["point_pct"] for x in per_run])),
                size_pct_median=float(np.median([x["size_pct"] for x in per_run])),
                losses_per_min_overall=sum(x["losses"] for x in per_run) / tot_min,
                refind_s=dict(n=int(len(refind)), same_frame=float(np.mean(refind == 0)), median=c.pct(refind, 50), p90=c.pct(refind, 90),
                              within_1s=float(np.mean(refind < 1.0))),
                putback_px=dict(n=int(len(putback)), median=c.pct(putback[:, 0], 50), p90=c.pct(putback[:, 0], 90),
                                matches_median=c.pct(putback[:, 1], 50)),
                size_noise_detrended=noise, loss_per_0_5s_window_by_max_body_rate=p_loss,
                loss_per_0_5s_window_by_max_body_rate_far=p_loss_far, far_means_width_below_px=FAR_PX)
    c.write_json("tracking_summary", summ)
    with open(f"{c.DATA}/runs_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0]))
        w.writeheader(); w.writerows(per_run)

    # --- figure: continuity + re-find + put-backs
    fig, axs = plt.subplots(1, 3, figsize=(c.COL2, 1.8))
    pr = sorted(per_run, key=lambda x: x["run"])
    axs[0].bar(range(len(pr)), [x["point_pct"] for x in pr], color=c.C["blue"], label="point sent")
    axs[0].plot(range(len(pr)), [x["size_pct"] for x in pr], "o", ms=2, color=c.C["orange"], label="size in packet")
    axs[0].set_ylabel("% of frames / packets")
    axs[0].set_xlabel("flight (chronological)")
    axs[0].set_ylim(0, 105)
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
    v = np.sort(np.maximum(refind, 0.04))
    axs[1].plot(v, np.arange(1, len(v) + 1) / len(v), color=c.C["green"], drawstyle="steps-post")
    axs[1].set_xscale("log")
    axs[1].set_xlim(0.03, None)
    axs[1].set_ylim(0, 1.02)
    axs[1].text(0.05, 0.40, f"{100 * np.mean(refind == 0):.0f} % same frame", fontsize=6.5, va="bottom")
    axs[1].set_xlabel("time to re-find a lost target [s]")
    axs[1].set_ylabel("CDF")
    axs[2].hist(putback[:, 0], bins=np.logspace(np.log10(8), np.log10(400), 25), color=c.C["red"])
    axs[2].set_xscale("log")
    axs[2].set_xlabel("put-back correction [px]")
    axs[2].set_ylabel("count")
    fig.tight_layout(w_pad=1.0)
    c.save(fig, "fig_tracking")

    # --- figure: size noise vs width (pixel values in tracking_summary.json)
    fig, ax = plt.subplots(figsize=(c.COL1, 1.9))
    wbins = np.array([5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 80, 150])
    mids, med, p90 = [], [], []
    for lo, hi in zip(wbins[:-1], wbins[1:]):
        m = (S[:, 0] >= lo) & (S[:, 0] < hi)
        if m.sum() >= 30:
            mids.append(np.sqrt(lo * hi)); med.append(100 * np.median(S[m, 1])); p90.append(100 * np.percentile(S[m, 1], 90))
    ax.plot(mids, med, "o-", color=c.C["blue"], ms=3, label="median [%]")
    ax.plot(mids, p90, "s--", color=c.C["orange"], ms=3, label="90th pct. [%]")
    ax.set_xscale("log")
    ax.set_xlabel("apparent branch width [px]")
    ax.set_ylabel("size noise [%]")
    ax.legend(loc="upper right", fontsize=6)
    c.save(fig, "fig_size_noise")

    # --- figure: loss probability vs angular rate (all windows, and only while still far from the branch)
    fig, ax = plt.subplots(figsize=(c.COL1, 1.8))
    for curve, col, mk, lab in ((p_loss, c.C["red"], "o-", "all tracked time"),
                                (p_loss_far, c.C["blue"], "s--", f"far only (width < {FAR_PX:.0f} px)")):
        pts = [b for b in curve if b["p_loss"] is not None]
        ax.plot([b["rate_median"] for b in pts], [100 * b["p_loss"] for b in pts], mk, color=col, ms=3, label=lab)
        if curve is p_loss:
            for b in pts:
                ax.annotate(f"{b['n']}", (b["rate_median"], 100 * b["p_loss"]), textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlabel("max. body rate |ω| in 0.5 s window [°/s]")
    ax.set_ylabel("P(target lost) [%]")
    ax.legend(loc="upper left", fontsize=6)
    c.save(fig, "fig_loss_vs_rate")
    print(summ)


if __name__ == "__main__":
    main()
