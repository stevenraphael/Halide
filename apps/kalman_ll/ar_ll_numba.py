#!/usr/bin/env python3
# FAIR compiled coarse-batch baseline for the ar_ll workload: Numba njit with
# prange OVER THE BATCH and a plain sequential scalar scan over time per series --
# the RIGHT way to compile a batched scan on CPU (each core owns whole series, no
# cross-core sync, cache-resident per-series state). This is the honest competitor
# for inductive Halide (JAX lax.scan parallelizes the wrong / no axis on CPU).
# Latent AR(2) + observation-noise Kalman log-likelihood.
import numpy as np, time
from numba import njit, prange

with open("params_ar.txt") as f:
    B, T, ti, tn, phi1, phi2, q, Rv = f.read().split()
B, T = int(B), int(T)
ti, tn, phi1, phi2, q, Rv = map(float, (ti, tn, phi1, phi2, q, Rv))

z = np.fromfile("z_ar.bin", dtype=np.float64).reshape(T, B).T   # (B, T), C-contig
z = np.ascontiguousarray(z)
ll_halide = np.fromfile("ll_ar.bin", dtype=np.float64)          # (B,)


@njit(parallel=True, fastmath=False, cache=True)
def batched_loglik(z, phi1, phi2, q, Rv):
    B, T = z.shape
    out = np.empty(B, dtype=np.float64)
    for b in prange(B):
        x0 = 0.0; x1 = 0.0
        P00 = 1.0; P01 = 0.0; P11 = 1.0
        ll = 0.0
        for t in range(1, T):
            Pp00 = phi1*phi1*P00 + 2.0*phi1*phi2*P01 + phi2*phi2*P11 + q
            Pp01 = phi1*P00 + phi2*P01
            Pp11 = P00
            S = Pp00 + Rv
            K0 = Pp00 / S; K1 = Pp01 / S
            xm0 = phi1*x0 + phi2*x1; xm1 = x0
            innov = z[b, t] - xm0
            ll += -0.5 * (innov*innov / S + np.log(S))
            x0 = xm0 + K0*innov
            x1 = xm1 + K1*innov
            P00 = (1.0 - K0)*Pp00
            P01 = (1.0 - K0)*Pp01
            P11 = Pp11 - K1*Pp01
        out[b] = ll
    return out


ll_nb = batched_loglik(z, phi1, phi2, q, Rv)   # compile + run

best = 1e18
for _ in range(5):
    t0 = time.perf_counter()
    batched_loglik(z, phi1, phi2, q, Rv)
    best = min(best, (time.perf_counter() - t0) * 1e3)

abs_err = np.max(np.abs(ll_nb - ll_halide))
rel_err = np.max(np.abs(ll_nb - ll_halide) / np.maximum(np.abs(ll_halide), 1.0))
print(f"Latent AR(2) + obs-noise log-likelihood  B={B} T={T}  (|LL| ~ {np.median(np.abs(ll_halide)):.0f})")
print(f"  Halide inductive (fold):        {ti:8.3f} ms")
print(f"  Halide non-inductive (mat.):    {tn:8.3f} ms")
print(f"  Numba njit prange-batch (CPU):  {best:8.3f} ms")
print(f"  err vs Halide LL: abs {abs_err:.3g}  rel {rel_err:.3g}  -> {'PASS' if rel_err < 1e-4 else 'FAIL'}")
print(f"  ratio: Numba/inductive {best/ti:.2f}x, Numba/non-inductive {best/tn:.2f}x")
