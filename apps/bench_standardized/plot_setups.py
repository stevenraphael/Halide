#!/usr/bin/env python3
"""Per-setup speedup summary: fold vs a baseline, geomean'd within each setup.

Instead of one dot per parameter point, this collapses each SWEEP to a single
geomean speedup, but keeps QUALITATIVELY-DIFFERENT setups within an app separate.
Qualitative axes (per the paper): core count, aloe image size, and the testing
regime (recurrence-length vs per-step-work vs batch vs arithmetic-intensity, etc).
Points that differ only in the swept parameter are averaged together.

Baseline selectable (default non-inductive; --baseline unfolded for the folding
ablation). Metric is median time; speedup = baseline/folded.

Usage:
  python3 plot_setups.py ~/logs/log5.txt ~/logs/log5aloe.txt -o setups.pdf
  python3 plot_setups.py --baseline unfolded ~/logs/log5.txt ... -o folding_setups.pdf
"""
import argparse
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_logs import app_of, classify, param_sig, ROW, CMD  # noqa: E402

# ---- publication theme: serif (Computer-Modern-like) to match a LaTeX paper,
# muted palette, hairline despined axes. Falls back cleanly if CM/LM absent.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "Nimbus Roman",
                   "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.6,
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#222222",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# a restrained, cohesive qualitative palette (muted, colour-blind-friendly-ish)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

APP_ORDER = ["viterbi", "kalman", "chebyshev", "ode", "prefixsum", "stereobm"]


def is_regime_header(line):
    if line.startswith("### ") and any(k in line for k in
                                       ("sweep", "aloe", "flip", "intensity")):
        return True
    return line.startswith("--- ") and line.rstrip().endswith("---")


def threads_of(cmd):
    m = re.search(r"HL_NUM_THREADS=(\d+)", cmd)
    return m.group(1) if m else "par"


def setup_of(app, header, cmd):
    """App-aware qualitative-setup label, or None to drop the point."""
    h = header.lower()
    nt = threads_of(cmd)
    if app == "viterbi":
        if "recurrence-length" in h:
            return "recurrence-len (1c)"
        if "state-count" in h:
            return "per-step-work"
    elif app == "kalman":
        if "recurrence-length" in h:
            return f"recurrence-len ({nt}c)"
        if "batch-size" in h:
            return "batch"
        if "arith" in h:
            return "arith-intensity"
    elif app == "chebyshev":
        if "per-step-work" in h:
            return "per-step-work"
        if "recurrence-length" in h:
            return "recurrence-len"
    elif app == "ode":
        if "recurrence-length" in h:
            return "recurrence-len"
        if "batch" in h:
            return "batch"
    elif app == "prefixsum":
        if h.startswith("---") and "consumer=" in h:
            cons = "mean" if "consumer=div" in h else "shift"
            return f"{cons} ({nt}c)"
    elif app == "stereobm":
        # image size is the qualitative core-regime here (full-res threads co-vary
        # with tilesize); winsize distinguishes the two full-res passes.
        img = "full-res" if "_large" in cmd or "full-res" in h else "small"
        wm = re.search(r"WINSIZE=(\d+)", cmd)
        return f"{img} w{wm.group(1)}" if wm else img
    return None


def collect(paths, baseline):
    # (app, setup) -> list of speedups
    setups = OrderedDict()
    for path in paths:
        header, cmd = "", None
        # first pass groups (app, param_sig) -> folded/base ms + its setup label
        groups = OrderedDict()
        for line in Path(path).read_text(errors="replace").splitlines():
            if is_regime_header(line):
                header = line
                continue
            m = CMD.match(line)
            if m:
                cmd = m.group(1)
                continue
            if cmd is None:
                continue
            app = app_of(cmd)
            if app is None or app not in APP_ORDER:
                continue
            m = ROW.match(line)
            if not m:
                continue
            cls = classify(m.group("label"), baseline)
            if cls is None:
                continue
            key = (app, param_sig(cmd))
            g = groups.setdefault(key, dict(app=app, setup=setup_of(app, header, cmd),
                                            ours=[], base=[]))
            g[cls].append(float(m.group("ms")))
        for g in groups.values():
            if not g["ours"] or not g["base"] or g["setup"] is None:
                continue
            sp = min(g["base"]) / min(g["ours"])   # median-time speedup
            setups.setdefault((g["app"], g["setup"]), []).append(sp)
    return setups


def collect_combined(paths):
    """(app, setup) -> list of (folded_speedup, unfolded_speedup), BOTH measured
    against the same non-inductive baseline. The two series share a baseline, so
    the gap between them is the contribution of storage folding on top of fusion."""
    setups = OrderedDict()
    for path in paths:
        header, cmd = "", None
        groups = OrderedDict()
        for line in Path(path).read_text(errors="replace").splitlines():
            if is_regime_header(line):
                header = line
                continue
            m = CMD.match(line)
            if m:
                cmd = m.group(1)
                continue
            if cmd is None:
                continue
            app = app_of(cmd)
            if app is None or app not in APP_ORDER:
                continue
            m = ROW.match(line)
            if not m:
                continue
            label = m.group("label").lower()
            key = (app, param_sig(cmd))
            g = groups.setdefault(key, dict(app=app, setup=setup_of(app, header, cmd),
                                            fold=[], unf=[], base=[]))
            if classify(m.group("label"), "noninductive") == "ours":
                g["fold"].append(float(m.group("ms")))
            elif "unfolded" in label:
                g["unf"].append(float(m.group("ms")))
            elif classify(m.group("label"), "noninductive") == "base":
                g["base"].append(float(m.group("ms")))
        for g in groups.values():
            if not (g["fold"] and g["unf"] and g["base"]) or g["setup"] is None:
                continue
            # Carry the best (min) time of each variant for this param point, so the
            # caller can either form a per-point speedup (real parameter sweeps) or
            # a best-time-across-points speedup (stereobm's tilesize scheduling knob).
            setups.setdefault((g["app"], g["setup"]), []).append(
                (min(g["fold"]), min(g["unf"]), min(g["base"])))
    return setups


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("-o", "--out", default="setups.pdf")
    ap.add_argument("--baseline", choices=["noninductive", "unfolded"],
                    default="noninductive")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dat", default=None,
                    help="also write a PGFPlots sidecar (tab-sep, one geomean "
                         "column per app) for fig_setups.tex")
    ap.add_argument("--combined-dat", default=None,
                    help="write a sidecar with BOTH folded and unfolded geomeans "
                         "(vs the non-inductive baseline) for fig_combined_setups.tex")
    args = ap.parse_args()

    if args.combined_dat:
        cs = collect_combined([Path(p) for p in args.logs])

        def speedups(app, v):
            """v = list of (foldms, unfms, basems), one per swept param point.
            stereobm's swept axis is tilesize -- a pure scheduling knob on the
            SAME problem -- so we compare each implementation at its own best
            tilesize (best-time-vs-best-time), giving one point, no whisker. Every
            other app sweeps a real parameter (different problems), so we form a
            per-point speedup and summarize the set with a geomean + min-max."""
            if app == "stereobm":
                fmin = min(f for f, u, b in v)
                umin = min(u for f, u, b in v)
                bmin = min(b for f, u, b in v)
                return [bmin / fmin], [bmin / umin]
            return ([b / f for f, u, b in v], [b / u for f, u, b in v])

        crows = [(app, s, *speedups(app, v)) for (app, s), v in cs.items()]
        crows.sort(key=lambda r: (APP_ORDER.index(r[0]), -geomean(r[2])))
        hdr = ["idx", "label",
               "fgm", "fep", "fem", "flx", "fvl",
               "ugm", "uep", "uem", "ulx", "uvl"]
        out = ["\t".join(hdr)]
        print(f"{'app':10s} {'setup':22s} {'fold':>6s} {'unfold':>7s} {'n':>3s}")
        def errs(vals, g):  # (plus, minus); nan when there is no spread (n=1)
            if max(vals) - min(vals) < 1e-9:
                return "nan", "nan"
            return f"{max(vals) - g:.4f}", f"{g - min(vals):.4f}"

        for i, (app, s, fs, us) in enumerate(crows):
            fg, ug = geomean(fs), geomean(us)
            fep, fem = errs(fs, fg)
            uep, uem = errs(us, ug)
            label = f"{app[:4]}. {s}".replace("recurrence-len", "rec-len")
            out.append("\t".join([
                str(i), label,
                f"{fg:.4f}", fep, fem,
                f"{max(fg, max(fs)):.4f}", f"{fg:.2f}$\\times$",
                f"{ug:.4f}", uep, uem,
                f"{max(ug, max(us)):.4f}", f"{ug:.2f}$\\times$"]))
            print(f"{app:10s} {s:22s} {fg:6.2f} {ug:7.2f} {len(fs):3d}")
        Path(args.combined_dat).write_text("\n".join(out) + "\n")
        print(f"\nwrote combined sidecar {args.combined_dat}  ({len(crows)} setups)")
        return

    setups = collect([Path(p) for p in args.logs], args.baseline)
    # order rows by app then descending geomean
    rows = [(app, s, geomean(v), min(v), max(v), len(v))
            for (app, s), v in setups.items()]
    rows.sort(key=lambda r: (APP_ORDER.index(r[0]), -r[2]))

    print(f"{'app':10s} {'setup':22s} {'geomean':>8s} {'min':>6s} {'max':>6s} {'n':>3s}")
    for app, s, gm, lo, hi, n in rows:
        print(f"{app:10s} {s:22s} {gm:8.2f} {lo:6.2f} {hi:6.2f} {n:3d}")

    if args.dat:
        gmcol = {a: "gm" + a[0].upper() for a in APP_ORDER}
        # lx = where the value label sits (just past the longer of bar / whisker);
        # vlabel = preformatted "g.gg x" so the figure needs no number formatting.
        hdr = (["idx", "label", "eplus", "eminus", "lx", "vlabel"]
               + [gmcol[a] for a in APP_ORDER])
        out = ["\t".join(hdr)]
        for i, (app, s, gm, lo, hi, n) in enumerate(rows):
            label = f"{app[:4]}. {s}".replace("recurrence-len", "rec-len")
            cells = {gmcol[a]: "nan" for a in APP_ORDER}
            cells[gmcol[app]] = f"{gm:.4f}"
            vlabel = f"{gm:.2f}$\\times$"
            out.append("\t".join(
                [str(i), label, f"{hi - gm:.4f}", f"{gm - lo:.4f}",
                 f"{max(gm, hi):.4f}", vlabel]
                + [cells[gmcol[a]] for a in APP_ORDER]))
        Path(args.dat).write_text("\n".join(out) + "\n")
        print(f"\nwrote sidecar {args.dat}")

    color = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(APP_ORDER)}
    n = len(rows)
    gms = [r[2] for r in rows]
    los = [r[3] for r in rows]
    his = [r[4] for r in rows]
    ys = list(range(n))

    fig, ax = plt.subplots(figsize=(7.6, 0.34 * n + 1.1))
    fig.subplots_adjust(left=0.30, right=0.985, bottom=0.11 + 0.9 / (0.34 * n + 1.1) * 0.0)

    # faint alternating background band per app group + rotated app name in the
    # left margin, so group labels never collide with the per-setup tick labels.
    i = 0
    while i < n:
        j = i
        while j < n and rows[j][0] == rows[i][0]:
            j += 1
        if (APP_ORDER.index(rows[i][0])) % 2 == 1:
            ax.axhspan(i - 0.5, j - 0.5, color="#000000", alpha=0.04, zorder=0)
        ax.text(-0.29, (i + j - 1) / 2, rows[i][0],
                transform=ax.get_yaxis_transform(), rotation=90,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color=color[rows[i][0]])
        i = j

    # parity reference
    ax.axvline(1.0, ls=(0, (4, 3)), lw=0.8, color="#888888", zorder=1)

    bars = ax.barh(ys, gms, color=[color[r[0]] for r in rows], height=0.62,
                   zorder=3, edgecolor="white", linewidth=0.5)
    # min-max whisker across the swept parameter, with small caps
    for yi, (lo, hi) in enumerate(zip(los, his)):
        if hi > lo * 1.001:
            ax.plot([lo, hi], [yi, yi], color="#33333388", lw=0.9, zorder=4,
                    solid_capstyle="butt")
            for xv in (lo, hi):
                ax.plot([xv, xv], [yi - 0.16, yi + 0.16], color="#33333388",
                        lw=0.9, zorder=4)
    # value label placed clear of both the bar end and the whisker
    for yi, (gm, hi) in enumerate(zip(gms, his)):
        ax.text(max(gm, hi) * 1.04, yi, f"{gm:.2f}×", va="center", ha="left",
                fontsize=7.8, color="#222222", zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(min(los) / 1.15, max(his) * 1.45)
    ticks = [0.5, 1, 2, 3, 5, 8]
    ticks = [t for t in ticks if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: (f"{v:g}×")))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(length=3, width=0.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    base = "non-inductive" if args.baseline == "noninductive" else "unfolded"
    ax.set_xlabel(f"speedup vs. {base}   (bar: geomean;  whisker: min–max across sweep)")
    if args.title:
        ax.set_title(args.title, fontsize=10.5, pad=8)
    fig.savefig(args.out)
    print(f"\nwrote {args.out}  ({n} setups across "
          f"{len({r[0] for r in rows})} apps)")


if __name__ == "__main__":
    main()
