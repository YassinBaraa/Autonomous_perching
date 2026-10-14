#!/usr/bin/env python3
"""Approach precision against motion capture (OptiTrack/VRPN in the flight bags).

For every climb (controller CLIMBING [-> STRIKE] -> next state):
  * apparent size s (UDP packet, ibvs/target_point.z) vs camera height z: 1/s = (z_b - z)/K, fitted
    per climb (K = focal length x branch diameter, z_b = height where the size would diverge);
  * time-to-contact from size alone (least-squares slope of ln s over 1 s, as the controller does)
    vs ground truth TTC = (z_ref - z)/v_z, with z_ref = contact height + delta, delta (camera-branch
    offset at contact) taken from the OTHER climbs (so the truth for a climb uses none of its size data);
  * outcomes: contact (PUSH) and PERCHED per climb attempt.
Figures fig_size_distance, fig_ttc, fig_approach_example; data approach_climbs.csv, approach_summary.json."""
import csv

import numpy as np

import common as c
from common import plt

WIN = 1.0   # s, same window as the controller's TTC fit


def climbs():
    out = []
    for b in c.bag_names():
        st = c.bag_states(b)
        tp, od = c.bag(b, "ibvs/target_point"), c.bag(b, "vrpn_client/estimated_odometry")
        if tp is None or od is None or not len(od["t"]):
            continue
        for k, (t0, s) in enumerate(st):
            if s != "CLIMBING":
                continue
            nxt = [(t1, s1) for t1, s1 in st[k + 1:] if s1 != "STRIKE"]
            if not nxt:
                continue
            t_end, outcome = nxt[0]
            perched = outcome == "PUSH" and any(s2 == "PERCHED" for _, s2 in st[k + 1:k + 5])
            strike = [t1 for t1, s1 in st[k + 1:] if s1 == "STRIKE" and t1 <= t_end]
            out.append(dict(bag=b, t0=t0, t_end=t_end, outcome=outcome, perched=perched, strike=strike[0] if strike else None,
                            tp=tp, od=od, cl=c.bag(b, "ibvs/closeness")))
    return out


def smooth_vz(t, z, win=0.25):
    """dz/dt by a local linear fit over +-win/2."""
    v = np.full_like(z, np.nan)
    j0 = 0
    for i in range(len(t)):
        a, b = np.searchsorted(t, t[i] - win / 2), np.searchsorted(t, t[i] + win / 2)
        if b - a >= 5:
            v[i] = np.polyfit(t[a:b] - t[i], z[a:b], 1)[0]
    return v


def main():
    cl = climbs()
    rows = []
    for x in cl:
        tp, od = x["tp"], x["od"]
        m = (tp["t"] >= x["t0"]) & (tp["t"] <= x["t_end"]) & (tp["z"] > 0)
        x["n"] = int(m.sum())
        x["zc"] = float(np.interp(x["t_end"], od["t"], od["pz"]))
        x["xy_c"] = (float(np.interp(x["t_end"], od["t"], od["px"])), float(np.interp(x["t_end"], od["t"], od["py"])))
        x["fit"] = None
        if x["n"] >= 30:
            ts, s = tp["stamp"][m], tp["z"][m]
            z = np.interp(ts, od["stamp"], od["pz"])
            A = np.vstack([z, np.ones_like(z)]).T
            coef = np.linalg.lstsq(A, 1 / s, rcond=None)[0]
            K, zb = -1 / coef[0], -coef[1] / coef[0]
            r2 = 1 - np.sum((1 / s - A @ coef) ** 2) / np.sum((1 / s - np.mean(1 / s)) ** 2)
            x["fit"] = dict(K=K, zb=zb, r2=r2, ts=ts, s=s, z=z)
        rows.append(x)
    good = [x for x in rows if x["outcome"] == "PUSH" and x["fit"] and x["fit"]["r2"] > 0.9]
    deltas = {id(x): x["fit"]["zb"] - x["zc"] for x in good}

    # time-to-contact: size-only vs motion capture
    ttc_pairs = []          # (ttc_gt, ttc_size, ttc_ctrl, d_true, climb idx)
    dist_err = []           # relative distance error from size with the climb's own K
    for gi, x in enumerate(good):
        others = [d for k, d in deltas.items() if k != id(x)]
        delta = float(np.median(others))
        zref = x["zc"] + delta
        f = x["fit"]
        ts, s = f["ts"], f["s"]
        od = x["od"]
        mo = (od["stamp"] >= ts[0] - 1) & (od["stamp"] <= ts[-1] + 1)
        vz = smooth_vz(od["stamp"][mo], od["pz"][mo])
        d_true = zref - f["z"]
        dist_err.extend(list((f["K"] / s - (f["zb"] - f["z"])) / (f["zb"] - f["z"])))
        cl_ = x["cl"]
        for i in range(len(ts)):
            a = np.searchsorted(ts, ts[i] - WIN)
            if i - a < 8:
                continue
            slope = np.polyfit(ts[a:i + 1] - ts[i], np.log(s[a:i + 1]), 1)[0]
            v = float(np.interp(ts[i], od["stamp"][mo], vz))
            if slope <= 1e-3 or v <= 0.08 or d_true[i] <= 0.02:
                continue
            ttc_size = 1.0 / slope
            # the window's fit describes its middle: compare with the truth half a window earlier
            tmid = ts[i] - WIN / 2
            zmid = float(np.interp(tmid, od["stamp"], od["pz"]))
            vmid = float(np.interp(tmid, od["stamp"][mo], vz))
            ttc_gt = (zref - zmid) / vmid if vmid > 0.08 else np.nan
            ttc_c = np.nan
            if cl_ is not None and len(cl_["t"]):
                j = np.searchsorted(cl_["t"], ts[i])
                if j < len(cl_["t"]):
                    ttc_c = cl_["y"][j]
            ttc_pairs.append((ttc_gt, ttc_size, ttc_c, d_true[i], gi))
    P = np.array(ttc_pairs, float)
    ok = np.isfinite(P[:, 0]) & (P[:, 0] > 0) & (P[:, 0] < 20)
    P = P[ok]
    rel = (P[:, 1] - P[:, 0]) / P[:, 0]
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 20)]
    by_bin = {}
    for lo, hi in bins:
        mb = (P[:, 0] >= lo) & (P[:, 0] < hi)
        if mb.sum() >= 5:
            by_bin[f"{lo}-{hi} s"] = dict(n=int(mb.sum()), median_rel_err=float(np.median(rel[mb])),
                                          median_abs_rel_err=float(np.median(np.abs(rel[mb]))), p90_abs_rel_err=float(np.percentile(np.abs(rel[mb]), 90)))
    dist_err = np.array(dist_err)

    # outcomes
    att = len(rows)
    contact = sum(1 for x in rows if x["outcome"] == "PUSH")
    perched = sum(1 for x in rows if x["perched"])
    xy = np.array([x["xy_c"] for x in rows if x["perched"]])
    summ = dict(climb_attempts=att, contact=contact, perched=perched,
                aborted={o: sum(1 for x in rows if x["outcome"] == o) for o in sorted({x["outcome"] for x in rows}) if o != "PUSH"},
                size_fit=dict(n_climbs=len(good), r2_median=float(np.median([x["fit"]["r2"] for x in good])),
                              r2_min=float(min(x["fit"]["r2"] for x in good)), K_median_px_m=float(np.median([x["fit"]["K"] for x in good])),
                              K_range=[float(min(x["fit"]["K"] for x in good)), float(max(x["fit"]["K"] for x in good))],
                              camera_branch_offset_at_contact_m=dict(median=float(np.median(list(deltas.values()))),
                                                                     iqr=[float(np.percentile(list(deltas.values()), 25)), float(np.percentile(list(deltas.values()), 75))])),
                distance_from_size_rel_err=dict(median_abs=float(np.median(np.abs(dist_err))), p90_abs=float(np.percentile(np.abs(dist_err), 90))),
                ttc=dict(n=int(len(P)), median_abs_rel_err=float(np.median(np.abs(rel))), by_true_ttc=by_bin),
                contact_xy_spread_m=dict(std_x=float(np.std(xy[:, 0])), std_y=float(np.std(xy[:, 1])), n=int(len(xy))) if len(xy) > 1 else None)
    c.write_json("approach_summary", summ)
    with open(f"{c.DATA}/approach_climbs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bag", "t_climb_s", "outcome", "perched", "n_size", "K_px_m", "z_branch_fit_m", "z_contact_m", "r2", "contact_x", "contact_y"])
        for x in rows:
            fit = x["fit"] or {}
            w.writerow([x["bag"], round(x["t0"] - x["od"]["t"][0], 1), x["outcome"], x["perched"], x["n"],
                        fit.get("K") and round(fit["K"], 3), fit.get("zb") and round(fit["zb"], 3), round(x["zc"], 3),
                        fit.get("r2") and round(fit["r2"], 4), round(x["xy_c"][0], 3), round(x["xy_c"][1], 3)])

    # --- figure: size vs distance to the branch (all good climbs)
    fig, ax = plt.subplots(figsize=(c.COL1, 2.2))
    cmap = plt.get_cmap("viridis")
    for gi, x in enumerate(good):
        f = x["fit"]
        d = f["zb"] - f["z"]
        ax.loglog(d[d > 0.01], f["s"][d > 0.01], ".", ms=1.5, color=cmap(gi / max(len(good) - 1, 1)), alpha=0.7)
    dd = np.logspace(np.log10(0.05), np.log10(3), 50)
    Km = summ["size_fit"]["K_median_px_m"]
    ax.loglog(dd, Km / dd, "k--", lw=0.8, label=f"$s = K/d$, median $K$ = {Km:.1f} px·m")
    ax.set_xlabel("distance camera–branch $d$ [m] (motion capture)")
    ax.set_ylabel("apparent width $s$ [px]")
    ax.legend(loc="upper right")
    c.save(fig, "fig_size_distance")

    # --- figure: TTC from size vs ground truth
    fig, ax = plt.subplots(figsize=(c.COL1, 2.4))
    sc = ax.scatter(P[:, 0], P[:, 1], s=2, c=P[:, 3], cmap="viridis", alpha=0.6, lw=0)
    lim = [0.2, 20]
    ax.plot(lim, lim, "k--", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true time-to-contact [s] (motion capture)")
    ax.set_ylabel("time-to-contact from size [s]")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("distance [m]")
    c.save(fig, "fig_ttc")

    # --- figure: one approach in detail (last flight of the campaign)
    ex = [x for x in good if x["bag"].startswith("2026-10-13-18-11")] or good[-1:]
    x = ex[0]
    od, tp = x["od"], x["tp"]
    t0 = x["t0"] - 3
    mo = (od["t"] >= t0) & (od["t"] <= x["t_end"] + 2)
    mt = (tp["t"] >= t0) & (tp["t"] <= x["t_end"] + 2) & (tp["z"] > 0)
    fig, axs = plt.subplots(2, 1, figsize=(c.COL1, 2.6), sharex=True)
    axs[0].plot(od["t"][mo] - x["t0"], od["pz"][mo], color=c.C["blue"], label="height (motion capture)")
    axs[0].axhline(x["zc"], color=c.C["grey"], lw=0.6, ls=":")
    axs[0].set_ylabel("height [m]")
    axs[0].axvspan(0, x["t_end"] - x["t0"], color=c.C["orange"], alpha=0.08, lw=0)
    axs[0].text(0.1, x["zc"] - 0.25, "CLIMBING", fontsize=6.5, color=c.C["orange"])
    axs[1].semilogy(tp["t"][mt] - x["t0"], tp["z"][mt], color=c.C["green"])
    axs[1].set_ylabel("apparent width [px]")
    axs[1].set_xlabel("time from start of climb [s]")
    axs[1].axvspan(0, x["t_end"] - x["t0"], color=c.C["orange"], alpha=0.08, lw=0)
    for a in axs:
        a.axvline(x["t_end"] - x["t0"], color=c.C["red"], lw=0.7)
    axs[0].text(x["t_end"] - x["t0"] + 0.1, 0.6, "contact", fontsize=6.5, color=c.C["red"])
    c.save(fig, "fig_approach_example")
    print(f"climbs {att}, contact {contact}, perched {perched}; fit R2 median {summ['size_fit']['r2_median']:.4f}; "
          f"K median {Km:.2f}; offset at contact {summ['size_fit']['camera_branch_offset_at_contact_m']}")
    print("distance rel err", summ["distance_from_size_rel_err"], "| TTC", summ["ttc"]["median_abs_rel_err"], by_bin)
    print("contact xy spread", summ["contact_xy_spread_m"])


if __name__ == "__main__":
    main()
