"""Optimized mainstream-library reference for BDF3: SUNDIALS CVODE (compiled
C, the industrial-standard adaptive-order/adaptive-step BDF implementation),
via scikits.odes. CVODE has no literal "force order exactly 3, fixed step"
mode -- only CVodeSetMaxOrd to CAP the adaptive order at <=3 (it still
freely adapts order 1/2/3 and step size within that cap based on its own
error estimates). That's the closest a real solver gets to "BDF3-like"; we
report it as exactly that, not as an identical algorithm to our fixed-step
BDF3.

Run with:
  LD_LIBRARY_PATH=$HOME/.local/lib <venv>/bin/python3 \
      apps/ode_implicit/ode_bdf3_cvode.py
"""
import time
import numpy as np
from scikits.odes import ode

with open("apps/ode_implicit/bdf3_params.txt") as f:
    B, T, mu, h, K, hal_ms = f.readline().split()
B, T, mu, h, K, hal_ms = int(B), int(T), float(mu), float(h), int(K), float(hal_ms)
y1_0 = np.fromfile("apps/ode_implicit/bdf3_y1_0.bin", np.float64)
y2_0 = np.fromfile("apps/ode_implicit/bdf3_y2_0.bin", np.float64)
obs = np.fromfile("apps/ode_implicit/bdf3_obs.bin", np.float64).reshape(B, T)

t_final = (T - 1) * h
t_eval = np.arange(T) * h


def rhs(t, y, ydot):
    ydot[0] = y[1]
    ydot[1] = mu * (1 - y[0] ** 2) * y[1] - y[0]


N_SAMPLE = min(64, B)
diffs = []
# Construct the CVODE solver ONCE and reuse it across trajectories (just
# calling .solve() again with a different initial condition each time).
# Building a fresh solver object per trajectory re-initializes the whole
# SUNDIALS context/CVode memory/linear-solver setup every call, which costs
# ~11ms on its own -- 60x more than the actual 131-step integration (~0.19ms)
# -- so it would grossly overstate CVODE's real per-trajectory cost.
solver = ode("cvode", rhs, old_api=False,
             lmm_type="BDF", nonlinsolver="newton",
             order=3,               # cap adaptive order at <=3 (BDF3-like)
             rtol=1e-6, atol=1e-9)
t0 = time.perf_counter()
for b in range(N_SAMPLE):
    sol = solver.solve(t_eval, np.array([y1_0[b], y2_0[b]], dtype=np.float64))
    y1_cvode = np.tanh(sol.values.y[:, 0])
    diffs.append(float(np.max(np.abs(y1_cvode - obs[b]))))
elapsed_sample = (time.perf_counter() - t0) * 1e3
extrapolated_full = elapsed_sample * (B / N_SAMPLE)

print(f"BDF3 implicit ODE (Van der Pol)  B={B} T={T} mu={mu} h={h}  (Halide used K={K} fixed Newton steps)")
print(f"  Halide inductive:                                {hal_ms:8.3f} ms")
print(f"  SUNDIALS CVODE (compiled C, BDF, max order<=3,")
print(f"    adaptive step + adaptive Newton), {N_SAMPLE} trajectories")
print(f"    (no native batching):                          {elapsed_sample:8.3f} ms")
print(f"    -> extrapolated to full B={B}:                  {extrapolated_full:8.3f} ms"
      f"  ({extrapolated_full / hal_ms:.0f}x slower than Halide)")
print(f"  max abs diff vs Halide's fixed-step BDF3:         {max(diffs):.3g}"
      f"  (CVODE still adapts order 1-3 and step size within the cap, so this"
      f" isn't the identical algorithm -- see note in this file's docstring)")
