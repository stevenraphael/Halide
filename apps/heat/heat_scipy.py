"""Validate the Halide AB2 heat solver against SciPy solve_ivp on the same
system dy/dt = A y. Reads heat_data.txt (A, y0, Halide final field)."""
import time
import numpy as np
from scipy.integrate import solve_ivp

with open("apps/heat/heat_data.txt") as f:
    D, Bv, T, dt, tfinal = f.readline().split()
    D, Bv, T = int(D), int(Bv), int(T)
    tfinal = float(tfinal)
    vals = np.array([float(x) for x in f.read().split()])

off = 0
A = vals[off:off + D * D].reshape(D, D); off += D * D
y0 = vals[off:off + D * Bv].reshape(Bv, D); off += D * Bv
hal = vals[off:off + D * Bv].reshape(Bv, D)

def run(method, **kw):
    max_rel, t0 = 0.0, time.perf_counter()
    for b in range(Bv):
        sol = solve_ivp(lambda t, y: A @ y, [0.0, tfinal], y0[b], method=method, **kw)
        ref = sol.y[:, -1]
        max_rel = max(max_rel, np.linalg.norm(hal[b] - ref) / (np.linalg.norm(ref) + 1e-30))
    return max_rel, (time.perf_counter() - t0) * 1e3

# Accurate reference (adaptive) and a fixed-#step run closer to AB2's work.
acc_rel, acc_ms = run("DOP853", rtol=1e-10, atol=1e-12)
rk_rel, rk_ms = run("RK45", rtol=1e-6, atol=1e-9)

print(f"Heat eq validation vs SciPy solve_ivp (tfinal={tfinal:.4f}, {Bv} columns)")
print(f"  Halide AB2 vs DOP853(rtol1e-10): rel L2 = {acc_rel:.3e}  -> "
      f"{'PASS' if acc_rel < 5e-3 else 'FAIL'}")
print(f"  SciPy DOP853 time: {acc_ms:8.1f} ms for {Bv} cols "
      f"({acc_ms/Bv:.2f} ms/col)")
print(f"  SciPy RK45   time: {rk_ms:8.1f} ms for {Bv} cols "
      f"({rk_ms/Bv:.2f} ms/col)")
