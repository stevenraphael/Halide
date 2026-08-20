"""Fair third-party reference (CPU) for the batched steady-state Kalman scan,
with several Numba variants to push it as fast as possible without a GPU:
  v1: manual i,j matvec loops (dot over contiguous A rows), prange over batch
  v2: xn = A @ xp (Numba's optimized matvec), prange over batch
  v3: batch-TILED -- process a contiguous tile of filters together so the inner
      ops vectorize across filters (like Halide's batch-vectorized schedule)."""
import time
import numpy as np
from numba import njit, prange

with open("apps/kalman_ss/params.txt") as f:
    N, B, T, hal_ms = f.readline().split()
N, B, T, hal_ms = int(N), int(B), int(T), float(hal_ms)
A = np.fromfile("apps/kalman_ss/A.bin", np.float32).reshape(N, N)
K = np.fromfile("apps/kalman_ss/K.bin", np.float32)
C = np.fromfile("apps/kalman_ss/C.bin", np.float32)
z = np.ascontiguousarray(np.fromfile("apps/kalman_ss/z.bin", np.float32).reshape(T, B).T)
haly = np.fromfile("apps/kalman_ss/y.bin", np.float32).reshape(B, T)

@njit(parallel=True, fastmath=True, cache=True)
def v1(A, K, C, z):
    N = A.shape[0]; B, T = z.shape
    y = np.empty((B, T), np.float32)
    for b in prange(B):
        xp = np.empty(N, np.float32); xn = np.empty(N, np.float32)
        for i in range(N): xp[i] = K[i] * z[b, 0]
        s = np.float32(0)
        for i in range(N): s += C[i] * xp[i]
        y[b, 0] = s
        for t in range(1, T):
            for i in range(N):
                acc = np.float32(0)
                for j in range(N): acc += A[i, j] * xp[j]
                xn[i] = acc + K[i] * z[b, t]
            s = np.float32(0)
            for i in range(N): s += C[i] * xn[i]
            y[b, t] = s
            xp, xn = xn, xp
    return y

@njit(parallel=True, fastmath=True, cache=True)
def v2(A, K, C, z):
    N = A.shape[0]; B, T = z.shape
    y = np.empty((B, T), np.float32)
    for b in prange(B):
        xp = K * z[b, 0]
        y[b, 0] = C @ xp
        for t in range(1, T):
            xp = A @ xp + K * z[b, t]
            y[b, t] = C @ xp
    return y

@njit(parallel=True, fastmath=True, cache=True)
def v3(A, K, C, z, VT):
    # Batch-tiled: X is (N, VT) for a tile of VT filters; inner loops vectorize
    # across the tile (contiguous filter axis), matching Halide's schedule.
    N = A.shape[0]; B, T = z.shape
    y = np.empty((B, T), np.float32)
    ntile = B // VT
    for bt in prange(ntile):
        b0 = bt * VT
        Xp = np.empty((N, VT), np.float32); Xn = np.empty((N, VT), np.float32)
        for i in range(N):
            for v in range(VT): Xp[i, v] = K[i] * z[b0 + v, 0]
        for v in range(VT):
            s = np.float32(0)
            for i in range(N): s += C[i] * Xp[i, v]
            y[b0 + v, 0] = s
        for t in range(1, T):
            for i in range(N):
                for v in range(VT): Xn[i, v] = K[i] * z[b0 + v, t]
                for j in range(N):
                    a = A[i, j]
                    for v in range(VT): Xn[i, v] += a * Xp[j, v]
            for v in range(VT):
                s = np.float32(0)
                for i in range(N): s += C[i] * Xn[i, v]
                y[b0 + v, t] = s
            Xp, Xn = Xn, Xp
    return y

def bench(fn):
    fn()  # compile + warm
    best = 1e18
    for _ in range(5):
        t0 = time.perf_counter(); r = fn(); best = min(best, (time.perf_counter() - t0) * 1e3)
    return best, r

t1, y1 = bench(lambda: v1(A, K, C, z))
t2, y2 = bench(lambda: v2(A, K, C, z))
t3, y3 = bench(lambda: v3(A, K, C, z, 8 if B % 8 == 0 else 1))

def rel(y): return float(np.max(np.abs(y - haly)) / (np.max(np.abs(haly)) + 1e-6))
print(f"Steady-state Kalman  N={N} B={B} T={T}")
print(f"  Halide inductive:                 {hal_ms:8.3f} ms")
print(f"  Numba v1 (manual matvec loops):   {t1:8.3f} ms  (rel {rel(y1):.1e})")
print(f"  Numba v2 (A @ xp dot):            {t2:8.3f} ms  (rel {rel(y2):.1e})")
print(f"  Numba v3 (batch-tiled SIMD):      {t3:8.3f} ms  (rel {rel(y3):.1e})")
