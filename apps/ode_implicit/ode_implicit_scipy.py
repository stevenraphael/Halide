"""Real mainstream-library reference for the implicit Van der Pol integrator:
scipy.integrate.solve_ivp with method='Radau', an implicit Runge-Kutta stiff
solver with genuinely ADAPTIVE step size and an adaptive (convergence-checked)
Newton solve per step -- the dynamic-loop-bound case Halide cannot express
(no data-dependent iteration counts). scipy has no native batching, so each
trajectory is solved with its own independent Python-level call; this is the
realistic cost of using an off-the-shelf adaptive stiff solver on a batch of
B trajectories, not an artificially slow strawman.

We only run a subsample of the B trajectories (scipy per-call overhead makes
running all 4096 impractical) and extrapolate linearly to the full batch,
noting the (lack of) batching explicitly rather than hiding it.
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


def vdp_jac(t, y):
    return [[0.0, 1.0],
            [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1 - y[0] ** 2)]]


N_SAMPLE = min(32, B)
rel_errs = []
t0 = time.perf_counter()
for b in range(N_SAMPLE):
    sol = solve_ivp(vdp, [0.0, t_final], [y1_0[b], y2_0[b]],
                     method="Radau", jac=vdp_jac, t_eval=t_eval,
                     rtol=1e-6, atol=1e-9)
    y1_scipy = np.tanh(sol.y[0])
    rel_errs.append(float(np.max(np.abs(y1_scipy - obs[b]))))
elapsed_sample = (time.perf_counter() - t0) * 1e3
extrapolated_full = elapsed_sample * (B / N_SAMPLE)

print(f"Implicit ODE (Van der Pol)  B={B} T={T} mu={mu} h={h}  (Halide used K={K} fixed Newton steps)")
print(f"  Halide inductive:                              {hal_ms:8.3f} ms")
print(f"  scipy Radau (adaptive step + adaptive Newton),")
print(f"    {N_SAMPLE} trajectories (no native batching): {elapsed_sample:8.3f} ms")
print(f"    -> extrapolated to full B={B}:                {extrapolated_full:8.3f} ms"
      f"  ({extrapolated_full / hal_ms:.0f}x slower than Halide)")
print(f"  max abs diff vs Halide's fixed-step trajectory: {max(rel_errs):.3g}"
      f"  (expected: Radau follows the TRUE solution; fixed-step backward"
      f" Euler accumulates its own truncation error, so some difference is"
      f" correct behavior, not a bug)")
