#!/usr/bin/env python3
# Full-trajectory Kalman FILTER baseline in simdkalman (its primary use):
# kf.compute(..., filtered=True) returns the filtered state mean for every t,
# batched over series. simdkalman materializes covariances internally and loops
# over timesteps in Python.
import numpy as np, time, simdkalman

with open("traj_params.txt") as f:
    B, T, tind, tmat, q0, q1, Rv = f.read().split()
B, T = int(B), int(T); tind, tmat, q0, q1, Rv = map(float, (tind, tmat, q0, q1, Rv))

z = np.fromfile("traj_z.bin", dtype=np.float64).reshape(T, B).T
lvl_h = np.fromfile("traj_level.bin", dtype=np.float64).reshape(T, B).T
trd_h = np.fromfile("traj_trend.bin", dtype=np.float64).reshape(T, B).T

kf = simdkalman.KalmanFilter(
    state_transition=[[1.0, 1.0], [0.0, 1.0]],
    process_noise=[[q0, 0.0], [0.0, q1]],
    observation_model=[[1.0, 0.0]],
    observation_noise=[[Rv]])

z_sk = z[:, 1:]
init_val = np.zeros((B, 2, 1)); init_cov = np.tile(np.eye(2)[None], (B, 1, 1))

def run():
    return kf.compute(z_sk, 0, filtered=True, smoothed=False,
                      initial_value=init_val, initial_covariance=init_cov)

r = run()
st = np.asarray(r.filtered.states.mean)      # (B, T-1, 2)
best = 1e18
for _ in range(3):
    t0 = time.perf_counter(); run(); best = min(best, (time.perf_counter() - t0) * 1e3)

el = np.max(np.abs(st[:, :, 0] - lvl_h[:, 1:])); et = np.max(np.abs(st[:, :, 1] - trd_h[:, 1:]))
print(f"Kalman full-trajectory filter (d=2)  B={B} T={T}")
print(f"  Halide inductive (fold P->2, x full):  {tind:8.3f} ms")
print(f"  Halide materialize (P full + x full):  {tmat:8.3f} ms")
print(f"  simdkalman (filtered=True, numpy):     {best:8.3f} ms")
print(f"  state err vs Halide: level {el:.3g} trend {et:.3g} -> {'PASS' if max(el,et)<1e-6 else 'FAIL'}")
print(f"  speedup: inductive {best/tind:.0f}x vs simdkalman")
