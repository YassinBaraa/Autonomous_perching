#!/usr/bin/env python3
"""Height held from the apparent size only (controller height_source 'size', bags 10-13 11:41 onward):
segments with a constant ln(size) reference (ibvs/size_height x) of at least 3 s, the motion-capture
height in each (std, peak-to-peak) and how well the vision error z = ln(s_ref/s) = ln(d/d_ref) explains
the mocap height change (-dh = d_ref (e^z - 1), d_ref fitted, r^2).
Data size_hold_segments.csv, size_hold_summary.json."""
import csv

import numpy as np

import common as c

MIN_S = 3.0


def segments(t, x):
    idx = np.flatnonzero(np.isfinite(x))
    if not len(idx):
        return []
    out, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1 or abs(x[b] - x[a]) > 1e-6 or t[b] - t[a] > 0.5:
            out.append((start, a))
            start = b
    out.append((start, idx[-1]))
    return [(a, b) for a, b in out if t[b] - t[a] >= MIN_S]


def main():
    rows = []
    for b in c.bag_names():
        sh = c.bag(b, "ibvs/size_height")
        od = c.bag(b, "vrpn_client/estimated_odometry")
        if sh is None or od is None or not len(sh["t"]) or not len(od["t"]):
            continue
        ap = c.bag_states(b, "ibvs/approach")
        t = sh["t"]
        for s0, s1 in segments(t, sh["x"]):
            m = (od["t"] >= t[s0]) & (od["t"] <= t[s1])
            h = od["pz"][m]
            st = [s for tt, s in ap if tt <= t[s0]]
            z = sh["z"][s0:s1 + 1]
            dh = -(np.interp(t[s0:s1 + 1], od["t"], od["pz"]) - od["pz"][m][0])
            e = np.expm1(z) - np.expm1(z[0])
            g = np.isfinite(e)
            k = float(np.sum(e[g] * dh[g]) / max(np.sum(e[g] ** 2), 1e-9))
            r2 = float(1 - np.var(dh[g] - k * e[g]) / max(np.var(dh[g]), 1e-12))
            rows.append(dict(bag=b, t_start=round(t[s0] - od["t"][0], 1), duration_s=round(t[s1] - t[s0], 1),
                             approach=st[-1] if st else "", h_std_cm=round(100 * float(np.std(h)), 1),
                             h_p2p_cm=round(100 * float(np.ptp(h)), 1), z_std=round(float(np.nanstd(z)), 4),
                             d_ref_fit_m=round(k, 2), r2=round(r2, 2)))
    with open(f"{c.DATA}/size_hold_segments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    hold = [x for x in rows if x["approach"] != "CLIMBING" and x["duration_s"] >= 5.0]
    climb = [x for x in rows if x["approach"] == "CLIMBING"]
    dur = np.array([x["duration_s"] for x in hold])
    hs = np.array([x["h_std_cm"] for x in hold])
    summ = dict(segments_hold=len(hold), hold_seconds=float(dur.sum()), bags=len({x["bag"] for x in rows}),
                h_std_cm_median=c.pct(hs, 50), h_std_cm_time_weighted=float(np.sum(hs * dur) / dur.sum()),
                h_p2p_cm_median=c.pct(np.array([x["h_p2p_cm"] for x in hold]), 50),
                vision_vs_mocap_r2_hold_median=c.pct(np.array([x["r2"] for x in hold]), 50),
                vision_vs_mocap_r2_climb=[x["r2"] for x in climb],
                note=("inside a hold the height moves a few cm, the size noise (~2 % of ~1 m) is of the same order, "
                      "so r^2 is low there; over a climb (tens of cm) the size follows the height (r^2 0.98)"))
    c.write_json("size_hold_summary", summ)
    print(summ)


if __name__ == "__main__":
    main()
