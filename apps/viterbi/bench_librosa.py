#!/usr/bin/env python3
"""Benchmark librosa.sequence.viterbi against the Halide inductive-func
Viterbi decoder in viterbi_bench.cpp, on the exact same generated data.

Forced single-threaded (numba/BLAS thread pools pinned to 1) to match the
Halide side's single-core, vectorized schedule.

NOTE: the `trans` buffer dumped by viterbi_bench.cpp is row-stochastic with
trans[r, c] = P(from r -> to c), which is exactly librosa's `transition[i, j]
= P(i->j)` convention. Pass it through unchanged -- do NOT transpose it here,
even though the Halide formula itself reads the buffer transposed
(trans(current, prev)) internally.
"""

import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import librosa


def load(path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=3)
        S, M, T = [int(x) for x in header]
        init = np.fromfile(f, dtype=np.float32, count=S)
        trans = np.fromfile(f, dtype=np.float32, count=S * S).reshape(S, S)
        emit = np.fromfile(f, dtype=np.float32, count=S * M).reshape(S, M)
        obs = np.fromfile(f, dtype=np.int32, count=T)
    return S, M, T, init, trans, emit, obs


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/viterbi_bench_data.bin"
    S, M, T, init, trans, emit, obs = load(data_path)

    # prob[s, t] = P(obs[t] | state s), shape (n_states, n_steps).
    prob = emit[:, obs].astype(np.float64)
    transition = trans.astype(np.float64)
    transition /= transition.sum(axis=1, keepdims=True)
    p_init = init.astype(np.float64)

    # Warm up numba JIT (first call pays compilation cost).
    path, _logp = librosa.sequence.viterbi(
        prob, transition, p_init=p_init, return_logp=True
    )

    trials = 5
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        path, _logp = librosa.sequence.viterbi(
            prob, transition, p_init=p_init, return_logp=True
        )
        dt = (time.perf_counter() - t0) * 1000.0
        best = min(best, dt)

    print(
        f"librosa   (S={S}, M={M}, T={T}): best of {trials} = {best:.3f} ms "
        f"({S * T / best / 1000.0:.2f} Mstates/s)"
    )

    halide_path_file = data_path + ".path"
    if os.path.exists(halide_path_file):
        halide_path = np.fromfile(halide_path_file, dtype=np.int32)
        n_mismatch = int(np.sum(halide_path != path))
        print(
            f"Path mismatch vs Halide: {n_mismatch} / {T} "
            f"({'OK, ties aside' if n_mismatch == 0 else 'DIFFERS'})"
        )


if __name__ == "__main__":
    main()
