"""Shared paths, IEEE figure style and loaders for the paper analysis scripts."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "/home/crow-black/Autonomous_perching"
PAPER = f"{ROOT}/paper"
DATA = f"{PAPER}/data"
FIG = f"{PAPER}/figures"
REC = f"{ROOT}/UDP_client/recordings"
os.makedirs(FIG, exist_ok=True)

COL1 = 3.5     # IEEE single column [in]
COL2 = 7.16    # IEEE double column [in]
C = dict(blue="#1f5fa8", orange="#d9772b", green="#2e8b57", red="#c0392b", grey="#7f8c8d", purple="#7d5ba6", brown="#8b5a2b")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.linewidth": 0.3,
    "grid.alpha": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png")
    plt.close(fig)
    print(f"  figure {name}")


def load_csv(path):
    """dict column -> np.array (float where every value parses, else object)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    head, body = rows[0], rows[1:]
    out = {}
    for j, h in enumerate(head):
        col = [r[j] if j < len(r) else "" for r in body]
        try:
            if any("_" in v for v in col):   # float() accepts "20261012_161153"; run names stay strings
                raise ValueError
            out[h] = np.array([float(v) if v not in ("", "None", "nan") else np.nan for v in col])
        except ValueError:
            out[h] = np.array(col, dtype=object)
    return out


def runs():
    return load_csv(f"{DATA}/vision/runs.csv")


def frames(run):
    return load_csv(f"{DATA}/vision/{run}_frames.csv")


def bag_names():
    return sorted(os.path.basename(p) for p in glob.glob(f"{DATA}/bags/*") if os.path.isdir(p))


def bag(name, topic):
    p = f"{DATA}/bags/{name}/{topic.replace('/', '__')}.csv"
    return load_csv(p) if os.path.exists(p) else None


def bag_states(name, topic="ibvs/state"):
    p = f"{DATA}/bags/{name}/{topic.replace('/', '__')}.csv"
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [(float(t), s) for t, s in list(csv.reader(f))[1:]]


def pair_runs_bags():
    """{vision run: bag} for runs whose wall-clock span overlaps a bag's span (same Pi clock)."""
    spans = {}
    for b in bag_names():
        od = bag(b, "ibvs/thrust") or bag(b, "mavros/battery")
        if od is None or not len(od["t"]):
            continue
        spans[b] = (od["t"][0], od["t"][-1])
    pairs = {}
    r = runs()
    for run in r["run"]:
        fr = frames(run)
        w = fr["wall"][np.isfinite(fr["wall"])]
        if not len(w):
            continue
        a, z = w[0], w[-1]
        best = max(spans.items(), key=lambda kv: min(z, kv[1][1]) - max(a, kv[1][0]), default=None)
        if best and min(z, best[1][1]) - max(a, best[1][0]) > 5.0:
            pairs[run] = best[0]
    return pairs


def write_json(name, obj):
    with open(f"{DATA}/{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else str(v))


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")
