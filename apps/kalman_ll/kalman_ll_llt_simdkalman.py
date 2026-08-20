#!/usr/bin/env python3
# Third-party reference: simdkalman on the SAME batch of series the Halide
# kalman_ll_llt benchmark scored, using the realistic n_states=2 local-linear-
# trend model. Loads z_llt.bin/ll_llt.bin/params_llt.txt, runs simdkalman's
# vectorized filter for the per-series log-likelihood, validates vs Halide, times.
import numpy as np, time, simdkalman

with open("params_llt.txt") as f:
    B, T, ti, tn, q0, q1, Rv = f.read().split()
B, T = int(B), int(T)
ti, tn, q0, q1, Rv = map(float, (ti, tn, q0, q1, Rv))

# z_llt.bin: column-major, b fastest (Halide Buffer<double> z(B,T)):
# flat[b + t*B] -> reshape (T,B) then transpose to (B,T).
z = np.fromfile("z_llt.bin", dtype=np.float64).reshape(T, B).T   # (B, T)
ll_halide = np.fromfile("ll_llt.bin", dtype=np.float64)          # (B,)

# Local-linear-trend: state = [level, trend].
kf = simdkalman.KalmanFilter(
    state_transition   = [[1.0, 1.0], [0.0, 1.0]],
    process_noise      = [[q0, 0.0], [0.0, q1]],
    observation_model  = [[1.0, 0.0]],
    observation_noise  = [[Rv]])

# Halide sums observations z[1..T-1], predicting from posterior[0]=(x0=0, P0=I),
# never observing z[0]. Match: start from (0, I) and feed z[:, 1:].
z_sk = z[:, 1:]                                                  # (B, T-1)
init_val = np.zeros((B, 2, 1))                                   # x0 = [0, 0]
init_cov = np.tile(np.eye(2)[None, :, :], (B, 1, 1))            # P0 = I2

def run():
    r = kf.compute(z_sk, 0, filtered=False, smoothed=False, gains=False,
                   log_likelihood=True,
                   initial_value=init_val, initial_covariance=init_cov)
    return r.log_likelihood

ll_sk = run()
best = 1e18
for _ in range(5):
    t0 = time.perf_counter(); run(); best = min(best, (time.perf_counter()-t0)*1e3)

abs_err = np.max(np.abs(ll_sk - ll_halide))
rel_err = np.max(np.abs(ll_sk - ll_halide) / np.maximum(np.abs(ll_halide), 1.0))
print(f"Kalman local-linear-trend log-likelihood  B={B} T={T}  (|LL| ~ {np.median(np.abs(ll_halide)):.0f})")
print(f"  Halide inductive (fold):     {ti:8.3f} ms")
print(f"  Halide non-inductive (mat.): {tn:8.3f} ms")
print(f"  simdkalman (numpy, vec):     {best:8.3f} ms")
print(f"  err vs Halide LL: abs {abs_err:.3g}  rel {rel_err:.3g}  -> {'PASS' if rel_err < 1e-4 else 'FAIL'}")
print(f"  speedup: inductive {best/ti:.0f}x, non-inductive {best/tn:.0f}x  vs simdkalman")
