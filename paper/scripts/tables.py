#!/usr/bin/env python3
"""Paper tables: one row per climb attempt (flights) and one row per code version (development history).
Climbs from the controller state in the bags (fig_approach.climbs), joined to the vision run recorded at
the same time. Writes data/tables/flights.csv, versions.csv and the LaTeX (booktabs) versions of both."""
import csv
import os

import numpy as np

import common as c
from fig_approach import climbs

T = f"{c.DATA}/tables"
# code versions in flight order: (vision commit, ibvs label as logged, short description of what changed)
VERSIONS = [
    ("accbfd3", "b5f758a*", "V1", "stored target, perch point kept until reset"),
    ("bf252f2", "7ef3c03*", "V2", "size self-heals, point held on its spot, features off crossings"),
    ("e6cc71b", "48e97d0*", "V3", "hold through motion blur, no lock at the image edge"),
    ("acc8538", "721759c", "V4", "lock only where size works, steadier size, features added"),
    ("acc8538", "721759c*", "V4+", "V4 + put-back checks under development"),
    ("2deff6f", "5bef38d*", "V5", "put-back within 1 s, locks off crossings, size re-find limit"),
]


def version_of(label):
    for vis, ib, short, _ in VERSIONS:
        if label.startswith(vis) and label.rstrip().endswith("ibvs " + ib):
            return short
    return "?"


def events_between(fr, a, b, name):
    w, ev = fr["wall"], fr["events"]
    return int(sum(name in str(ev[i]).split("|") for i in np.flatnonzero((w >= a) & (w <= b))))


def main():
    os.makedirs(T, exist_ok=True)
    r = c.runs()
    pairs = c.pair_runs_bags()
    bag_runs = {}
    for run, b in pairs.items():
        bag_runs.setdefault(b, []).append(run)
    run_ver = {run: version_of(str(v)) for run, v in zip(r["run"], r["version"])}

    rows = []
    for x in climbs():
        st = c.bag_states(x["bag"])
        align = [t for t, s in st if s == "ALIGN" and t <= x["t0"]]
        runs_here = [run for run in bag_runs.get(x["bag"], [])
                     if (lambda f: len(f["wall"]) and f["wall"][0] <= x["t0"] <= f["wall"][-1])(c.frames(run))]
        run = runs_here[0] if runs_here else None
        row = dict(bag=x["bag"], run=run or "", version=run_ver.get(run, "pre-log"),
                   align_s=round(x["t0"] - align[-1], 1) if align else None, climb_s=round(x["t_end"] - x["t0"], 1),
                   outcome=x["outcome"], contact=x["outcome"] == "PUSH", perched=bool(x["perched"]))
        if run:
            fr = c.frames(run)
            lock = [fr["wall"][i] for i in range(len(fr["wall"])) if "LOCKED" in str(fr["events"][i]).split("|") and fr["wall"][i] <= x["t0"]]
            row["lock_to_climb_s"] = round(x["t0"] - lock[-1], 1) if lock else None
            m = (fr["wall"] >= x["t0"]) & (fr["wall"] <= x["t_end"])
            row["point_pct_in_climb"] = round(100 * float(np.mean(np.isfinite(fr["px"][m]))), 1) if m.any() else None
            row["size_pct_in_climb"] = round(100 * float(np.mean(np.isfinite(fr["size"][m]))), 1) if m.any() else None
            row["losses_in_climb"] = events_between(fr, x["t0"], x["t_end"], "LOST")
            row["putbacks_in_climb"] = events_between(fr, x["t0"], x["t_end"], "PUT BACK")
        rows.append(row)
    keys = ["bag", "run", "version", "align_s", "lock_to_climb_s", "climb_s", "outcome", "contact", "perched",
            "point_pct_in_climb", "size_pct_in_climb", "losses_in_climb", "putbacks_in_climb"]
    with open(f"{T}/flights.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})

    # per version
    with open(f"{c.DATA}/runs_metrics.csv") as f:
        rm = {x["run"]: x for x in csv.DictReader(f)}
    vrows = []
    for vis, ib, short, desc in [("—", "—", "pre-log", "10-12 morning flights, no vision log")] + VERSIONS:
        runs_v = [run for run in r["run"] if run_ver[run] == short]
        cl = [x for x in rows if x["version"] == short]
        if not runs_v and not cl:
            continue
        tm = sum(float(rm[run]["tracking_min"]) for run in runs_v)
        n_loss = sum(int(rm[run]["losses"]) for run in runs_v)
        n_pb = sum(int(rm[run]["putbacks"]) for run in runs_v)
        sel = np.isin(r["run"], runs_v)
        vrows.append(dict(version=short, commits=f"{vis} / {ib}", change=desc, runs=len(runs_v), tracking_min=round(tm, 1),
                          point_pct=round(float(np.median(r["point_pct"][sel])), 1) if runs_v else None,
                          size_pct=round(float(np.median(r["size_pct"][sel])), 1) if runs_v else None,
                          losses_per_min=round(n_loss / tm, 2) if tm else None, putbacks_per_min=round(n_pb / tm, 2) if tm else None,
                          climbs=len(cl), contact=sum(x["contact"] for x in cl), perched=sum(x["perched"] for x in cl)))
    with open(f"{T}/versions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vrows[0]))
        w.writeheader(); w.writerows(vrows)

    def tex(v):
        return "--" if v is None or v == "" else str(v).replace("_", r"\_").replace("%", r"\%")
    with open(f"{T}/versions.tex", "w") as f:
        f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\n"
                "Ver. & Change & Runs & Point [\\%] & Losses/min & Climbs & Contact & Perched \\\\\n\\midrule\n")
        for v in vrows:
            f.write(" & ".join(tex(v[k]) for k in ("version", "change", "runs", "point_pct", "losses_per_min", "climbs", "contact", "perched"))
                    + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(f"{T}/flights.tex", "w") as f:
        f.write("\\begin{tabular}{llrrrll}\n\\toprule\n"
                "Flight & Ver. & Align [s] & Lock$\\to$climb [s] & Climb [s] & Outcome & Perched \\\\\n\\midrule\n")
        for x in rows:
            f.write(" & ".join([tex(f"{x['bag'][5:10]} {x['bag'][11:13]}:{x['bag'][14:16]}"), tex(x["version"]), tex(x["align_s"]), tex(x.get("lock_to_climb_s")),
                                tex(x["climb_s"]), tex(x["outcome"]), "yes" if x["perched"] else "no"]) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for v in vrows:
        print(v)
    print(len(rows), "climbs")
    for x in rows:
        print({k: x.get(k) for k in keys})


if __name__ == "__main__":
    main()
