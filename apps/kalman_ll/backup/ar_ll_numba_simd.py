#!/usr/bin/env python3
# SIMD-over-batch Numba baseline for ar_ll -- the strongest fair CPU competitor.
# Unlike prange-over-batch (one SCALAR series per core, thread-parallel only),
# this gives Numba the SAME parallelism shape as Halide: thread-parallel over
# batch CHUNKS, and a vectorizable inner loop over the series WITHIN a chunk so
# LLVM emits SIMD across the batch. Requires the batch axis to be CONTIGUOUS:
# Halide dumps z(b,t) at offset b + t*B (t-major), so reshape(T,B) has b
# contiguous per time row -- exactly what the inner j-loop needs to vectorize.
# Latent AR(2) + observation-noise Kalman log-likelihood.
import numpy as np, time
from numba import njit, prange

with open("params_ar.txt") as f:
    B, T, ti, tn, phi1, phi2, q, Rv = f.read().split()
B, T = int(B), int(T)
ti, tn, phi1, phi2, q, Rv = map(float, (ti, tn, phi1, phi2, q, Rv))

zTB = np.fromfile("z_ar.bin", dtype=np.float64).reshape(T, B)   # (T,B), b contiguous
ll_halide = np.fromfile("ll_ar.bin", dtype=np.float64)          # (B,)

CH = 512  # series per thread-chunk; inner loop over the chunk is SIMD-vectorized


@njit(parallel=True, fastmath=False, cache=True)
def batched_loglik(zTB, phi1, phi2, q, Rv, CH):
    T, B = zTB.shape
    out = np.empty(B, dtype=np.float64)
    nchunks = (B + CH - 1) // CH
    for ci in prange(nchunks):
        lo = ci * CH
        hi = lo + CH
        if hi > B:
            hi = B
        n = hi - lo
        x0 = np.zeros(n); x1 = np.zeros(n)
        P00 = np.ones(n); P01 = np.zeros(n); P11 = np.ones(n)
        ll = np.zeros(n)
        for t in range(1, T):
            zt = zTB[t]                       # contiguous length-B row
            for j in range(n):                # vectorizable across batch
                b = lo + j
                Pp00 = phi1*phi1*P00[j] + 2.0*phi1*phi2*P01[j] + phi2*phi2*P11[j] + q
                Pp01 = phi1*P00[j] + phi2*P01[j]
                Pp11 = P00[j]
                S = Pp00 + Rv
                K0 = Pp00 / S; K1 = Pp01 / S
                xm0 = phi1*x0[j] + phi2*x1[j]; xm1 = x0[j]
                innov = zt[b] - xm0
                ll[j] += -0.5 * (innov*innov / S + np.log(S))
                x0[j] = xm0 + K0*innov
                x1[j] = xm1 + K1*innov
                P00[j] = (1.0 - K0)*Pp00
                P01[j] = (1.0 - K0)*Pp01
                P11[j] = Pp11 - K1*Pp01
        for j in range(n):
            out[lo + j] = ll[j]
    return out


ll_nb = batched_loglik(zTB, phi1, phi2, q, Rv, CH)   # compile + run

best = 1e18
for _ in range(5):
    t0 = time.perf_counter()
    batched_loglik(zTB, phi1, phi2, q, Rv, CH)
    best = min(best, (time.perf_counter() - t0) * 1e3)

abs_err = np.max(np.abs(ll_nb - ll_halide))
rel_err = np.max(np.abs(ll_nb - ll_halide) / np.maximum(np.abs(ll_halide), 1.0))
print(f"Latent AR(2) + obs-noise log-likelihood  B={B} T={T}  (|LL| ~ {np.median(np.abs(ll_halide)):.0f})")
print(f"  Halide inductive (fold):        {ti:8.3f} ms")
print(f"  Halide non-inductive (mat.):    {tn:8.3f} ms")
print(f"  Numba SIMD-over-batch (CPU):    {best:8.3f} ms")
print(f"  err vs Halide LL: abs {abs_err:.3g}  rel {rel_err:.3g}  -> {'PASS' if rel_err < 1e-4 else 'FAIL'}")
print(f"  ratio: Numba/inductive {best/ti:.2f}x, Numba/non-inductive {best/tn:.2f}x")
