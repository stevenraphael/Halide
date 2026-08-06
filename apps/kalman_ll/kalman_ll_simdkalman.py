#!/usr/bin/env python3
# Third-party reference: simdkalman on the SAME batch of series the Halide
# kalman_ll benchmark scored. Loads z.bin/ll.bin/params.txt, runs simdkalman's
# vectorized Kalman filter to get the per-series log-likelihood, validates
# against the Halide result, and times it.
import time

import numpy as np
import simdkalman

with open("params.txt") as f:
    B, T, ti, tn, F, H, Qv, Rv = f.read().split()
B, T = int(B), int(T)
ti, tn, F, H, Qv, Rv = map(float, (ti, tn, F, H, Qv, Rv))

# z.bin is column-major with b fastest (Halide Buffer<double> z(B,T)):
# flat[b + t*B] -> reshape (T, B) then transpose to (B, T).
z = np.fromfile("z.bin", dtype=np.float64).reshape(T, B).T  # (B, T)
ll_halide = np.fromfile("ll.bin", dtype=np.float64)  # (B,)

kf = simdkalman.KalmanFilter(
    state_transition=[[F]],
    process_noise=[[Qv]],
    observation_model=[[H]],
    observation_noise=[[Rv]],
)

# Halide's LL sums observations z[1..T-1], predicting from posterior[0]=(x0,P0)
# and never observing z[0]. Match exactly: start simdkalman from (x0,P0) and feed
# it z[:, 1:] (predict-update over exactly those T-1 observations).
z_sk = z[:, 1:]  # (B, T-1)
init_val = np.zeros((B, 1, 1))  # x0 = 0
init_cov = np.tile(np.array([[[1.0]]]), (B, 1, 1))  # P0 = 1


def run():
    r = kf.compute(
        z_sk,
        0,
        filtered=False,
        smoothed=False,
        gains=False,
        log_likelihood=True,
        initial_value=init_val,
        initial_covariance=init_cov,
    )
    return r.log_likelihood


ll_sk = run()  # warm up / correctness
best = 1e18
for _ in range(5):
    t0 = time.perf_counter()
    run()
    best = min(best, (time.perf_counter() - t0) * 1e3)

abs_err = np.max(np.abs(ll_sk - ll_halide))
rel_err = np.max(np.abs(ll_sk - ll_halide) / np.maximum(np.abs(ll_halide), 1.0))
print(
    f"Kalman log-likelihood  B={B} T={T}  (|LL| ~ {np.median(np.abs(ll_halide)):.0f})"
)
print(f"  Halide inductive (fold):     {ti:8.3f} ms")
print(f"  Halide non-inductive (mat.): {tn:8.3f} ms")
print(f"  simdkalman (numpy, vec):     {best:8.3f} ms")
print(
    f"  err vs Halide LL: abs {abs_err:.3g}  rel {rel_err:.3g}  -> {'PASS' if rel_err < 1e-4 else 'FAIL'}"
)
print(
    f"  speedup: inductive {best / ti:.0f}x, non-inductive {best / tn:.0f}x  vs simdkalman"
)
