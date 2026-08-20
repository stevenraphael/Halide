#!/usr/bin/env python3
"""Memory-footprint plot: inductive-fold vs non-inductive recurrence-state bytes.

One panel per app; for each parameter setting a thin stem connects the
non-inductive footprint (top) to the inductive-fold footprint (bottom), so the
vertical gap IS the storage-folding win. Log-scale y (bytes).

Reads the same logs as parse_logs.py and reuses its variant classification. It
prefers the BYTE-EXACT `state_bytes=<N>` field (emitted by the updated harness /
run_mem.sh); if absent (older logs) it falls back to the rounded `state_MB`
column -- which floors sub-10KB and zero footprints to 0, so a log plot from old
logs is only a preview. Re-run with run_mem.sh for the real figure.

Usage:  python3 plot_memory.py ~/logs/*.txt -o memory.pdf     # or a single file
"""
import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_logs import app_of, param_sig, CMD  # noqa: E402


def mem_classify(label):
    """Memory-plot classification. The baseline is the Halide NON-INDUCTIVE
    formulation that MATERIALIZES the recurrence state (so its footprint is in
    Halide's allocator and comparable). External references (OpenCV, Boost,
    oracle) and vendor scans report 0 / out-of-allocator bytes, so they are NOT
    memory baselines. Unfolded-inductive is also excluded (still inductive)."""
    l = label.lower()
    if "unfolded" in l:
        return None
    if "inductive folded" in l or "reg fold" in l or "one-stage folded" in l:
        return "ours"
    if ("materialize" in l or "schedule4" in l or "rdom, materialize" in l
            or "non-inductive" in l):
        return "base"
    return None

# state_MB column (the %10.2f before PASS/UNCHECKED/FAIL) and optional exact bytes
MEM = re.compile(r"^ {2}(?P<label>\S.*?)\s{2,}\d+\.\d+\s+\d+\.\d+\s+[\d.]+\s+\S+/s"
                 r"\s+(?P<mb>[\d.]+)\s+(?:PASS|FAIL|UNCHECKED)")
BYTES = re.compile(r"state_bytes=(\d+)")

APP_ORDER = ["viterbi", "kalman", "chebyshev", "ode", "prefixsum", "stereobm"]


def collect(paths):
    # (app, sig) -> dict(ours=[bytes], base=[bytes])
    groups = OrderedDict()
    for path in paths:
        cur = None
        for line in Path(path).read_text(errors="replace").splitlines():
            m = CMD.match(line)
            if m:
                cur = m.group(1)
                continue
            if cur is None:
                continue
            app = app_of(cur)
            if app is None or app not in APP_ORDER:
                continue
            m = MEM.match(line)
            if not m:
                continue
            cls = mem_classify(m.group("label"))
            if cls is None:
                continue
            mb = BYTES.search(line)
            b = float(mb.group(1)) if mb else float(m.group("mb")) * 1024 * 1024
            g = groups.setdefault((app, param_sig(cur)),
                                  dict(app=app, ours=[], base=[]))
            g[cls].append(b)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("-o", "--out", default="memory.pdf")
    args = ap.parse_args()

    groups = collect(args.logs)
    per_app = OrderedDict((a, []) for a in APP_ORDER)
    for g in groups.values():
        if not g["ours"] or not g["base"]:
            continue
        per_app[g["app"]].append((min(g["ours"]), min(g["base"])))

    per_app = OrderedDict((a, v) for a, v in per_app.items() if v)
    names = list(per_app)
    n = len(names)
    if not n:
        sys.exit("no paired (inductive, non-inductive) memory points found")

    # log floor for zero-byte footprints (drawn at the floor with a marker)
    allb = [x for v in per_app.values() for pair in v for x in pair if x > 0]
    floor = min(allb) / 4 if allb else 1.0

    fig, ax = plt.subplots(figsize=(1.5 * n + 1.5, 4.4))
    rng = np.random.default_rng(0)

    def clamp(b):
        return b if b > 0 else floor

    for i, name in enumerate(names):
        for (ours, base) in per_app[name]:
            x = i + (rng.random() - 0.5) * 0.5
            ax.plot([x, x], [clamp(ours), clamp(base)], color="0.7", lw=0.8, zorder=1)
            ax.scatter([x], [clamp(base)], s=40, color="tab:orange", zorder=3,
                       marker=("v" if base > 0 else "x"))
            ax.scatter([x], [clamp(ours)], s=40, color="tab:blue", zorder=3,
                       marker=("^" if ours > 0 else "x"))

    ax.set_yscale("log")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("recurrence-state footprint (bytes)")
    ax.set_xlim(-0.6, n - 0.4)
    for i in range(1, n):
        ax.axvline(i - 0.5, color="0.92", lw=0.8, zorder=0)
    if any(x == 0 for v in per_app.values() for pair in v for x in pair):
        ax.axhline(floor, ls=":", lw=0.8, color="0.6")
    ax.set_title("Recurrence-state footprint: inductive fold vs non-inductive")
    ax.scatter([], [], marker="v", color="tab:orange", label="non-inductive (materialize)")
    ax.scatter([], [], marker="^", color="tab:blue", label="inductive fold")
    ax.scatter([], [], marker="x", color="0.4", label="zero state (at floor)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}  ({n} apps, "
          f"{sum(len(v) for v in per_app.values())} paired points)")


if __name__ == "__main__":
    main()
