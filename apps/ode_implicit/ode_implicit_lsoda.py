"""Optimized mainstream-library reference for the implicit Van der Pol
integrator: scipy.integrate.solve_ivp with method='LSODA', which wraps
ODEPACK's compiled Fortran LSODA -- the same industrial-strength adaptive
stiff/nonstiff-switching solver used across engineering/scientific computing
for decades. Unlike the pure-Python 'Radau' method in scipy (see
ode_implicit_scipy.py, ~36000x slower than Halide), this is genuinely
optimized compiled code, so it isolates "adaptive step + adaptive Newton
inside an optimized library" from "slow because unvectorized Python."

Still has genuinely dynamic loop bounds (adaptive step size, internal Newton
iterations run to convergence) and no native batching -- each trajectory is
its own solve_ivp call.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp

with open("apps/ode_implicit/params.txt") as f:
    B, T, mu, h, K, hal_ms = f.readline().split()
B, T, mu, h, K, hal_ms = int(B), int(T), float(mu), float(h), int(K), float(hal_ms)
y1_0 = np.fromfile("apps/ode_implicit/y1_0.bin", np.float32)
y2_0 = np.fromfile("apps/ode_implicit/y2_0.bin", np.float32)
obs = np.fromfile("apps/ode_implicit/obs.bin", np.float32).reshape(B, T)

t_final = (T - 1) * h
t_eval = np.arange(T) * h


def vdp(t, y):
    return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]


N_SAMPLE = min(128, B)
diffs = []
t0 = time.perf_counter()
for b in range(N_SAMPLE):
    sol = solve_ivp(vdp, [0.0, t_final], [y1_0[b], y2_0[b]],
                     method="LSODA", t_eval=t_eval, rtol=1e-6, atol=1e-9)
    y1_lsoda = np.tanh(sol.y[0])
    diffs.append(float(np.max(np.abs(y1_lsoda - obs[b]))))
elapsed_sample = (time.perf_counter() - t0) * 1e3
extrapolated_full = elapsed_sample * (B / N_SAMPLE)

print(f"Implicit ODE (Van der Pol)  B={B} T={T} mu={mu} h={h}  (Halide used K={K} fixed Newton steps)")
print(f"  Halide inductive:                               {hal_ms:8.3f} ms")
print(f"  scipy LSODA (compiled ODEPACK, adaptive step +")
print(f"    adaptive internal Newton), {N_SAMPLE} trajectories")
print(f"    (no native batching):                         {elapsed_sample:8.3f} ms")
print(f"    -> extrapolated to full B={B}:                 {extrapolated_full:8.3f} ms"
      f"  ({extrapolated_full / hal_ms:.0f}x slower than Halide)")
print(f"  max abs diff vs Halide's fixed-step trajectory:  {max(diffs):.3g}"
      f" (measures fixed-step truncation error, not a bug -- see note in"
      f" ode_implicit_scipy.py)")
