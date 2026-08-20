#!/usr/bin/env python3
"""Parse the bench-harness logs in ~/logs into results.csv for plot_speedup.py.

Two log conventions are handled:
  * CPU logs: several variants printed inside ONE `=== cmd ===` block
    ("non-inductive (materialize)", "inductive UNFOLDED ...", "inductive FOLDED ...").
  * GPU logs: the inductive and non-inductive versions are SEPARATE binaries
    (separate blocks) grouped under a `########## ... ##########` param cell
    (prefixsum_gpu vs prefixsum_gpu_thrust; mamba_gpu_batched vs mamba_v1_bench).

Per group we take:
    ours     = best throughput among INDUCTIVE-FOLDED variants
    baseline = best throughput among genuinely NON-INDUCTIVE variants
Unfolded-inductive variants are EXCLUDED (an unfolded inductive func is still
inductive, not a non-inductive baseline). If a group has an inductive result but
no non-inductive one that ran, baseline is emitted as OOM/FAIL.

speedup is left to plot_speedup.py; we always emit metric=throughput.
"""
import re
import sys
from pathlib import Path

LOGDIR = Path.home() / "logs"

# --- classification of a variant label -------------------------------------
OURS = ("inductive folded", "reg fold", "halide 2-stage", "one-stage folded")
# genuinely non-inductive == a different FORMULATION that materializes / slides a
# dense window, NOT a hand-written parallel scan. (oneTBB/Thrust/CUB/mamba-v1 are
# all associative *scans* -> excluded: a scan is inductive, just not ours.)
# The baseline is the genuinely non-inductive Halide FORMULATION (materialize /
# RDom-slide / schedule4). External references (OpenCV, Boost.odeint, the C++
# oracle) are a DIFFERENT algorithm/implementation, reported separately -- they
# are NOT the non-inductive baseline, so they must not be classified here (else
# min(base) can pick the faster external ref, e.g. OpenCV over stereobm/schedule4).
NONIND = ("non-inductive", "rdom, materialize", "schedule4", "materialize")
# never a baseline and never "ours"
EXCLUDE = ("unfolded", "peak-ref", "unsegmented", "inclusivesum")


def classify(label, baseline="noninductive"):
    """Classify a variant row as our folded implementation ("ours"), the chosen
    baseline ("base"), or irrelevant (None). Two baseline modes:
      noninductive: base = a genuinely non-inductive FORMULATION (default plot).
      unfolded:     base = the same inductive func with folding OFF -- isolates
                    storage folding from fusion (the folding ablation)."""
    l = label.lower()
    is_ours = any(k in l for k in OURS)
    if baseline == "unfolded":
        if is_ours:
            return "ours"
        if "unfolded" in l:
            return "base"
        return None
    if any(k in l for k in EXCLUDE):
        return None
    if is_ours:
        return "ours"
    if any(k in l for k in NONIND):
        return "base"
    return None


BIN2APP = {
    "viterbi_log": "viterbi", "viterbi": "viterbi",
    "kalman_ar": "kalman",
    "chebyshev_inductive": "chebyshev",
    "ode_observer_sparse_fused_test": "ode", "ode_observer_sparse_test": "ode",
    "stereobm_jit": "stereobm",
    # mamba + GPU-prefixsum binaries are intentionally ABSENT: mamba has no
    # non-inductive counterpart, and prefixsum uses its CPU data only (the GPU
    # baseline is Thrust, a vendor scan -- not a non-inductive formulation).
    "prefixsum_bench": "prefixsum", "prefixsum_bench_fold": "prefixsum",
    "prefixsum_bench_rdom": "prefixsum", "prefixsum_bench_onestage": "prefixsum",
    "prefixsum_bench_tbb": "prefixsum", "prefixsum_sweep": "prefixsum",
}


def app_of(text):
    """App is the ./binary name; unknown/omitted binaries -> None (skipped)."""
    if text:
        for b in re.findall(r"\./([a-z_0-9]+)", text):
            if b in BIN2APP:
                return BIN2APP[b]
    return None


def param_sig(cmd):
    """Parameter signature: the operating point, independent of which binary/
    variant produced it, so cross-binary runs (prefixsum) pair with within-block
    ones (viterbi). Drops binary name, tmp paths, and fold/backend toggles."""
    s = re.sub(r"/\S*tmp\S+", "", cmd)          # tmp paths
    s = re.sub(r"\./[a-z_0-9]+", "", s)          # binary name
    s = re.sub(r"\b(numactl|--cpunodebind=0|--membind=0|env|UNFOLD=1|USE_GPU=1)\b",
               "", s)                             # backend/fold toggles (not a param)
    return re.sub(r"\s+", " ", s).strip()


# a variant row: "  <label>   <best_ms> <median_ms> <NNN.N Unit>  ..."
# We report MEDIAN (2nd column): best-of-N is optimistic and understates tail
# variance; median is the defensible headline.
ROW = re.compile(r"^ {2}(?P<label>\S.*?)\s{2,}"
                 r"\d+\.\d+\s+(?P<ms>\d+\.\d+)\s+"
                 r"(?P<tput>[\d.]+)\s+\S+/s")
CMD = re.compile(r"^=== (.+?) ===")
CELL = re.compile(r"^#{6,}\s+(\S.*?)\s+#{6,}\s*$")


def parse(path, baseline="noninductive"):
    """Yield (app, param, ours_ms, base_ms_or_None, base_failed).

    Groups by (app, parameter-signature) so runs pair regardless of whether the
    variants share one `=== cmd ===` block (viterbi: several variants per block)
    or are separate binaries at the same operating point (prefixsum: folded vs
    RDom-materialize as distinct invocations). ms is the common currency --
    throughput columns are not comparable across binaries.
    """
    groups = {}  # (app, sig) -> dict(app, param, ours=[], base=[], failed=bool)
    cur_cmd = None
    for line in path.read_text(errors="replace").splitlines():
        m = CMD.match(line)
        if m:
            cur_cmd = m.group(1)
            continue
        if cur_cmd is None:
            continue
        app = app_of(cur_cmd)
        if app is None:            # unknown/omitted binary (mamba, GPU prefixsum)
            continue
        key = (app, param_sig(cur_cmd))
        g = groups.setdefault(key, dict(app=app, param=param_sig(cur_cmd),
                                        ours=[], base=[], failed=False))
        if line.startswith("!!! FAILED") or "EXCEPTION" in line:
            g["failed"] = True     # this operating point had a run that couldn't launch
            continue
        m = ROW.match(line)
        if not m:
            continue
        cls = classify(m.group("label"), baseline)
        if cls is None:
            continue
        g[cls].append(float(m.group("ms")))

    for g in groups.values():
        if not g["ours"]:
            continue
        ours = min(g["ours"])            # fastest inductive-folded at this point
        if g["base"]:
            yield g["app"], clean(g["param"]), ours, min(g["base"]), False
        elif g["failed"]:
            yield g["app"], clean(g["param"]), ours, None, True


def clean(p):
    p = re.sub(r"/tmp\S+", "", p)          # drop tmp paths
    p = re.sub(r"\s+", " ", p).strip()
    return p[:60]


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Parse bench logs -> speedup CSV (fold vs a chosen baseline).")
    ap.add_argument("--baseline", choices=["noninductive", "unfolded"],
                    default="noninductive",
                    help="noninductive: vs a non-inductive formulation (default). "
                         "unfolded: vs the same inductive func with folding off "
                         "(the storage-folding ablation).")
    ap.add_argument("logs", nargs="*", help="log files (default: ~/logs/*.txt)")
    args = ap.parse_args()

    files = [Path(p) for p in args.logs] if args.logs else sorted(LOGDIR.glob("*.txt"))
    out = []
    for f in files:
        if f.name.endswith("Zone.Identifier"):
            continue
        for app, param, ours, base, failed in parse(f, args.baseline):
            base_field = "OOM" if base is None else f"{base:.4f}"
            out.append((app, param, "time", f"{ours:.4f}", base_field))

    # stable app order for the panels. mamba is EXCLUDED from the non-inductive
    # plot (no non-inductive counterpart -- materializing all of h is ~8 GiB); in
    # the unfolded ablation its GPU UNFOLD run OOMs, so it drops out there too.
    order = ["viterbi", "kalman", "chebyshev", "ode", "prefixsum", "stereobm"]
    out = [r for r in out if r[0] in order]
    out.sort(key=lambda r: (order.index(r[0]), r[1]))

    print("app,param,metric,inductive,noninductive")
    for r in out:
        print(",".join(r))
    print(f"# {len(out)} rows across {len({r[0] for r in out})} apps "
          f"(baseline={args.baseline})", file=sys.stderr)


if __name__ == "__main__":
    main()
