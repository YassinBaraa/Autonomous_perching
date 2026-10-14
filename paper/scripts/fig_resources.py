#!/usr/bin/env python3
"""CPU, memory, temperature and SoC power per mode (from resource_run.py).
Figure fig_resources; data resources_summary.json."""
import numpy as np

import common as c
from common import plt

D = f"{c.DATA}/resources"


def main():
    s = c.load_csv(f"{D}/samples.csv")
    rep = c.load_csv(f"{D}/modes_replay.csv")
    det = c.load_csv(f"{D}/modes_detector_only.csv")
    ph = s["phase"]
    wall = s["wall"]
    # mode of each replay sample: share of tracking frames in the 0.25 s before it
    mode = np.array(["idle" if p == "idle" else ("detector" if p == "detector_only" else "") for p in ph], dtype=object)
    for i in np.flatnonzero(ph == "replay"):
        m = (rep["wall"] > wall[i] - 0.25) & (rep["wall"] <= wall[i])
        if m.sum():
            mode[i] = "tracking" if rep["tracking"][m].mean() > 0.5 else "searching"
    summ = {}
    for md in ("idle", "searching", "tracking", "detector"):
        m = mode == md
        if not m.any():
            continue
        summ[md] = dict(samples=int(m.sum()), proc_cpu_pct=c.pct(s["proc_cpu_pct"][m], 50), sys_cpu_pct=c.pct(s["sys_cpu_pct"][m], 50),
                        rss_mb=c.pct(s["rss_mb"][m], 50), rss_max_mb=float(np.nanmax(s["rss_mb"][m])) if md != "idle" else None,
                        soc_power_w=c.pct(s["soc_power_w"][m], 50), temp_c=c.pct(s["temp_c"][m], 50))
    # CPU time per frame (independent of how fast the replay ran): process CPU seconds / frames, per mode
    for md, src in (("searching", rep), ("tracking", rep), ("detector", det)):
        cpu_s, n = 0.0, 0
        for i in np.flatnonzero(mode == md):
            if not np.isfinite(s["proc_cpu_pct"][i]):
                continue
            m = (src["wall"] > wall[i] - 0.25) & (src["wall"] <= wall[i])
            cpu_s += s["proc_cpu_pct"][i] / 100.0 * 0.25
            n += int(m.sum())
        if md in summ and n:
            summ[md]["cpu_ms_per_frame"] = 1000.0 * cpu_s / n
    t_rep = rep["wall"]
    summ["replay_fps"] = dict(all=float((len(t_rep) - 1) / (t_rep[-1] - t_rep[0])),
                              tracking_median_period_ms=float(1000 * np.median(np.diff(t_rep)[rep["tracking"][1:] > 0])))
    summ["detector_only_fps"] = float((len(det["wall"]) - 1) / (det["wall"][-1] - det["wall"][0]))
    summ["note"] = ("proc_cpu_pct: 100 % = one of the 4 Cortex-A76 cores; soc_power_w: sum of the Pi 5 PMIC rails "
                    "(SoC, RAM, IO), not the USB camera or the Hailo HAT")
    c.write_json("resources_summary", summ)

    t0 = wall[0]
    fig, axs = plt.subplots(3, 1, figsize=(c.COL1, 3.0), sharex=True)
    for ax, key, lab in zip(axs, ("proc_cpu_pct", "rss_mb", "soc_power_w"), ("CPU [%]", "RSS [MB]", "power [W]")):
        y = s[key]
        ok = np.isfinite(y)
        ax.plot(wall[ok] - t0, y[ok], lw=0.7, color=c.C["blue"])
        ax.set_ylabel(lab)
        for md, col in (("searching", c.C["orange"]), ("tracking", c.C["green"]), ("detector", c.C["red"])):
            m = mode == md
            ax.scatter(wall[m & ok] - t0, y[m & ok], s=2.5, color=col, zorder=3, label=md if key == "proc_cpu_pct" else None)
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, markerscale=2)
    fig.align_ylabels(axs)
    axs[-1].set_xlabel("time [s]  (idle → flight replay → detector only)")
    c.save(fig, "fig_resources")
    for k, v in summ.items():
        print(k, v)


if __name__ == "__main__":
    main()
