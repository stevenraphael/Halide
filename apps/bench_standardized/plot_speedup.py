#!/usr/bin/env python3
"""Speedup strip plot: 7 app panels, one dot per swept parameter setting.

Each dot is the speedup of the INDUCTIVE (fold) implementation over the
NON-INDUCTIVE (unfold / dense / third-party-serial) baseline for one parameter
setting. y=1.0 is parity (dashed line); >1 means inductive wins.

Input CSV schema (header required), one row per (app, parameter setting):
    app,param,metric,inductive,noninductive
  - app          : short app name; rows are grouped into panels by this, in
                   first-seen order (so order your CSV to control panel order).
  - param        : label for the swept point (e.g. "T=8192" or "L=64,B=8").
                   Only used in hover/annotation; not drawn, keeps dots anonymous.
  - metric       : "throughput" (higher is better) or "time" (lower is better).
                   Determines how speedup is computed from the two columns.
  - inductive    : the inductive/fold measurement (throughput or time).
  - noninductive : the non-inductive baseline measurement, OR one of the
                   sentinels {OOM, FAIL, NA} when the baseline could not run
                   (e.g. unfold exceeds the 2^31 buffer cap / won't launch).
                   Those points are drawn at the top as upward carets, since
                   "inductive runs where the baseline cannot" is itself a result.

speedup = inductive/noninductive        (metric=throughput)
        = noninductive/inductive        (metric=time)

Usage:
    python3 plot_speedup.py results.csv -o speedup.pdf
"""
import argparse
import csv
import math
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SENTINELS = {"OOM", "FAIL", "NA", "-"}


def load(path):
    apps = OrderedDict()  # app -> list of (speedup or None, param, sentinel or None)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            app = row["app"].strip()
            param = row.get("param", "").strip()
            metric = row["metric"].strip().lower()
            ind = float(row["inductive"])
            nb = row["noninductive"].strip()
            apps.setdefault(app, [])
            if nb.upper() in SENTINELS:
                apps[app].append((None, param, nb.upper()))
                continue
            nb = float(nb)
            if metric.startswith("t") and metric != "time":  # throughput
                sp = ind / nb
            elif metric == "throughput":
                sp = ind / nb
            elif metric == "time":
                sp = nb / ind
            else:
                sys.exit(f"unknown metric {metric!r} (use throughput|time)")
            apps[app].append((sp, param, None))
    return apps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default="speedup.pdf")
    ap.add_argument("--title", default="Inductive fold vs non-inductive baseline")
    ap.add_argument("--ylabel", default="speedup  (inductive / non-inductive)")
    args = ap.parse_args()

    apps = load(args.csv)
    names = list(apps.keys())
    n = len(names)

    # global y-range from real (non-sentinel) speedups
    allsp = [s for pts in apps.values() for (s, _, _) in pts if s is not None]
    if not allsp:
        sys.exit("no numeric speedups in CSV")
    lo, hi = min(allsp), max(allsp)
    top = hi * 1.6  # where sentinel carets sit

    fig, ax = plt.subplots(figsize=(1.4 * n + 1.5, 4.2))
    rng = np.random.default_rng(0)
    for i, name in enumerate(names):
        pts = apps[name]
        xs, ys = [], []
        for (s, _param, sent) in pts:
            x = i + (rng.random() - 0.5) * 0.5  # jitter within the panel column
            if s is None:
                ax.scatter([x], [top], marker="^", s=55, color="tab:red",
                           zorder=3, clip_on=False)
            else:
                xs.append(x)
                ys.append(s)
        ax.scatter(xs, ys, s=42, alpha=0.8, color="tab:blue",
                   edgecolor="white", linewidth=0.4, zorder=3)

    ax.axhline(1.0, ls="--", lw=1, color="0.4", zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(args.ylabel)
    ax.set_xlim(-0.6, n - 0.4)
    for i in range(1, n):
        ax.axvline(i - 0.5, color="0.9", lw=0.8, zorder=0)
    ax.set_title(args.title)
    # legend for the sentinel marker
    ax.scatter([], [], marker="^", color="tab:red", label="baseline OOM/can't run")
    ax.scatter([], [], color="tab:blue", label="one parameter setting")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({n} apps, "
          f"{sum(len(v) for v in apps.values())} points)")


if __name__ == "__main__":
    main()
