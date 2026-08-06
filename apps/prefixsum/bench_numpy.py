#!/usr/bin/env python3
"""Benchmark an optimized numpy implementation of the prefix-sum-then-average
pipeline from tutorial/lesson_25_inductive.cpp against the Halide inductive
schedule in prefixsum_bench.cpp, on the exact same generated data.

Pipeline: input(x, y) = x + y
          prefix_sum(x, y) = sum_{i<=x} input(i, y)
          output(x, y) = prefix_sum(x, y) // (x + 1)

Forced single-threaded (BLAS/OMP pinned to 1) to match Halide's single-core
schedule; numpy's cumsum is memory-bound, not compute-bound, so threading
doesn't help it much anyway.
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _fused_prefix_mean(inp, out):
    H, W = inp.shape
    for y in range(H):
        acc = np.int32(0)
        for x in range(W):
            acc = np.int32(acc + inp[y, x])
            out[y, x] = acc // np.int32(x + 1)


def compute_numba(inp):
    out = np.empty_like(inp)
    _fused_prefix_mean(inp, out)
    return out


def load(path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        W, H = int(header[0]), int(header[1])
        inp = np.fromfile(f, dtype=np.int32, count=W * H).reshape(H, W)
        halide_out = np.fromfile(f, dtype=np.int32, count=W * H).reshape(H, W)
    return W, H, inp, halide_out


def compute(inp, W):
    # Accumulate in int32, matching Halide's Int(32) prefix_sum exactly
    # (including silent wraparound for large W where the running sum
    # exceeds INT32_MAX) -- using int64 here would disagree with Halide
    # near the high end of each row instead of reproducing its overflow.
    # Halide's `/` on signed integers is floor division (like Python's //),
    # not C's truncating division, so numpy's // matches even when the
    # wrapped prefix sum goes negative.
    counts = np.arange(1, W + 1, dtype=np.int32)
    return np.cumsum(inp, axis=1, dtype=np.int32) // counts


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prefixsum_bench_data.bin"
    W, H, inp, halide_out = load(data_path)

    out = compute(inp, W)  # warm-up (also correctness reference below).

    trials = 10
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        out = compute(inp, W)
        dt = (time.perf_counter() - t0) * 1000.0
        best = min(best, dt)

    print(
        f"numpy     (W={W}, H={H}): best of {trials} = {best:.3f} ms "
        f"({W * H / best / 1000.0:.2f} Mpixels/s)"
    )

    n_mismatch = int(np.sum(out != halide_out))
    print(
        f"Output mismatch vs Halide: {n_mismatch} / {W * H} "
        f"({'OK' if n_mismatch == 0 else 'DIFFERS'})"
    )

    # numba: a fused, single-pass JIT'd loop with one int32 accumulator per
    # row -- structurally the same as Halide's compute_at(x).store_at(y)
    # .fold_storage(x, 1) schedule (no materialized prefix_sum array),
    # instead of numpy's two-pass cumsum-then-divide.
    out_nb = compute_numba(inp)  # warm-up (pays numba JIT compile cost).

    best_nb = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        out_nb = compute_numba(inp)
        dt = (time.perf_counter() - t0) * 1000.0
        best_nb = min(best_nb, dt)

    print(
        f"numba     (W={W}, H={H}): best of {trials} = {best_nb:.3f} ms "
        f"({W * H / best_nb / 1000.0:.2f} Mpixels/s)"
    )

    n_mismatch_nb = int(np.sum(out_nb != halide_out))
    print(
        f"Output mismatch vs Halide: {n_mismatch_nb} / {W * H} "
        f"({'OK' if n_mismatch_nb == 0 else 'DIFFERS'})"
    )


if __name__ == "__main__":
    main()
