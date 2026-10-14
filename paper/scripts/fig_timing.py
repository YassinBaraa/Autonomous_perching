#!/usr/bin/env python3
"""Throughput and latency: frame period by mode (searching = detector runs, tracking = KLT only),
per-stage times from the flight logs, and end-to-end latency camera grab -> UDP send -> attitude
command (flight bags). Figures fig_frame_period, fig_latency; data timing_summary.json."""
import numpy as np

import common as c
from common import plt


def main():
    r = c.runs()
    per_mode = {"searching": [], "tracking": []}
    for run in r["run"]:
        fr = c.frames(run)
        t = fr["t"]
        st = fr["tstate"]
        dt = np.diff(t) * 1000.0
        mode = np.array(["tracking" if s == "tracking" else "searching" for s in st[:-1]])
        for m in per_mode:
            per_mode[m].append(dt[mode == m])
    pm = {m: np.concatenate(v) for m, v in per_mode.items()}
    pm = {m: v[(v > 0) & (v < 2000)] for m, v in pm.items()}

    # stage timings from the logs (frames where the detector ran / KLT-only frames)
    det = c.load_csv(f"{c.DATA}/vision/timing_detection.csv")
    ib = c.load_csv(f"{c.DATA}/vision/timing_ibvs.csv")
    mn = c.load_csv(f"{c.DATA}/vision/timing_main.csv")
    keep = det["read"] < 1000   # first frame of a run includes opening the camera
    stages = {k: det[k][keep] for k in ("read", "detect", "track", "postprocess", "total")}
    trk = ib["source_read"] < 20   # raw camera read = tracking frame
    tracking_frame = ib["frame_total"][trk]

    # end-to-end latency from the flight bags
    ages, pickup, total = [], [], []
    for run, b in c.pair_runs_bags().items():
        fr = c.frames(run)
        a = fr["age"][np.isfinite(fr["age"])] * 1000.0
        ages.append(a)
        tp, sp = c.bag(b, "ibvs/target_point"), c.bag(b, "mavros/setpoint_raw/attitude")
        if tp is None or sp is None or not len(tp["t"]):
            continue
        j = np.searchsorted(sp["t"], tp["t"])
        ok = j < len(sp["t"])
        pickup.append((sp["t"][j[ok]] - tp["t"][ok]) * 1000.0)
        total.append((sp["t"][j[ok]] - tp["stamp"][ok]) * 1000.0)
    ages, pickup, total = (np.concatenate(x) for x in (ages, pickup, total))
    total = total[(total > 0) & (total < 500)]

    summ = dict(
        frame_period_ms={m: dict(n=len(v), median=c.pct(v, 50), p90=c.pct(v, 90), p99=c.pct(v, 99), rate_hz=1000.0 / c.pct(v, 50)) for m, v in pm.items()},
        detection_frame_stages_ms={k: dict(median=c.pct(v, 50), p10=c.pct(v, 10), p90=c.pct(v, 90)) for k, v in stages.items()},
        detection_frames=int(keep.sum()), tracking_frames=int(trk.sum()),
        tracking_frame_ms=dict(median=c.pct(tracking_frame, 50), p90=c.pct(tracking_frame, 90)),
        main_loop_overhead_ms=dict(median=c.pct(mn["loop_total"], 50), p99=c.pct(mn["loop_total"], 99)),
        latency_ms=dict(grab_to_udp=dict(n=len(ages), median=c.pct(ages, 50), p90=c.pct(ages, 90), p99=c.pct(ages, 99)),
                        udp_to_command=dict(n=len(pickup), median=c.pct(pickup, 50), p90=c.pct(pickup, 90)),
                        grab_to_command=dict(n=len(total), median=c.pct(total, 50), p90=c.pct(total, 90), p99=c.pct(total, 99))))
    c.write_json("timing_summary", summ)

    # --- figure: frame period CDF by mode
    fig, ax = plt.subplots(figsize=(c.COL1, 1.9))
    for m, col, lab, xy, ha in (("tracking", c.C["blue"], "tracking (KLT)", (0.03, 0.80), "left"),
                                ("searching", c.C["orange"], "searching (detector)", (0.97, 0.35), "right")):
        v = np.sort(pm[m])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col)
        ax.text(*xy, f"{lab}\nmedian {np.median(v):.0f} ms", color=col, fontsize=6.5, ha=ha, transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xlabel("frame period [ms]")
    ax.set_ylabel("CDF")
    ax.set_xlim(15, 1000)
    c.save(fig, "fig_frame_period")

    # --- figure: latency CDFs
    fig, ax = plt.subplots(figsize=(c.COL1, 1.9))
    for v, col, lab in ((ages, c.C["green"], "grab → UDP send"), (pickup, c.C["grey"], "UDP → attitude cmd"),
                        (total, c.C["red"], "grab → attitude cmd")):
        v = np.sort(v[np.isfinite(v)])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, label=f"{lab}, median {np.median(v):.0f} ms")
    ax.set_xlabel("latency [ms]")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 120)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=6.5)
    c.save(fig, "fig_latency")
    print({m: (round(summ['frame_period_ms'][m]['median'], 1), round(summ['frame_period_ms'][m]['p90'], 1)) for m in pm})
    print("stages", {k: round(v['median'], 1) for k, v in summ['detection_frame_stages_ms'].items()})
    print("latency", {k: (round(v['median'], 1), round(v.get('p90', 0), 1)) for k, v in summ['latency_ms'].items()})


if __name__ == "__main__":
    main()
