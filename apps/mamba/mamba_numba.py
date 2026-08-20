"""Fair third-party CPU reference for the Mamba selective-SSM scan: Numba
@njit(parallel=True) with prange over the channel axis (the coarse-grained
compiled-scan strategy, same as the Halide schedule). This is the honest
"hand-tuned compiled scan" bar the inductive Halide version should match."""
import time
import numpy as np
from numba import njit, prange

with open("apps/mamba/params.txt") as f:
    N, D, T, hal_ms = f.readline().split()
N, D, T, hal_ms = int(N), int(D), int(T), float(hal_ms)
# Dumped d-innermost: a(d,n,t) -> contiguous reshape (T,N,D); b,c (T,N); x (T,D).
a = np.fromfile("apps/mamba/a.bin", np.float32).reshape(T, N, D)
b = np.fromfile("apps/mamba/b.bin", np.float32).reshape(T, N)
c = np.fromfile("apps/mamba/c.bin", np.float32).reshape(T, N)
x = np.fromfile("apps/mamba/x.bin", np.float32).reshape(T, D)
haly = np.fromfile("apps/mamba/y.bin", np.float32).reshape(D, T)


@njit(parallel=True, fastmath=True, cache=True)
def scan(a, b, c, x):
    T, N, D = a.shape
    y = np.empty((D, T), np.float32)
    for d in prange(D):
        h = np.empty(N, np.float32)
        s = np.float32(0)
        for n in range(N):
            h[n] = b[0, n] * x[0, d]
            s += c[0, n] * h[n]
        y[d, 0] = s
        for t in range(1, T):
            xd = x[t, d]
            s = np.float32(0)
            for n in range(N):
                h[n] = a[t, n, d] * h[n] + b[t, n] * xd
                s += c[t, n] * h[n]
            y[d, t] = s
    return y


def parallel_scan(a, b, c, x):
    # SOTA algorithm class: the diagonal linear recurrence is an ASSOCIATIVE scan,
    # so Mamba's real kernel (selective_scan / Mamba-2 SSD) parallelizes over TIME
    # via a work-efficient prefix scan, not a sequential per-channel loop. Here:
    # Hillis-Steele inclusive scan of the affine maps h -> A*h + B, vectorized
    # across all (n,d) lanes. Combine(L,R) = (A_L*A_R, A_R*B_L + B_R), L earlier.
    A = a.copy()                                 # (T,N,D)
    B = (b[:, :, None] * x[:, None, :]).astype(np.float32)   # (T,N,D)
    d = 1
    while d < T:
        Ar = A[d:]; Br = B[d:]                    # right segments (t = d..T-1)
        Al = A[:T - d]; Bl = B[:T - d]            # left  segments (t-d)
        B[d:] = Ar * Bl + Br
        A[d:] = Al * Ar
        d *= 2
    # h_t = B-component of the inclusive prefix (h_{-1}=0). y[d,t]=sum_n c[t,n] h.
    y = np.einsum('tn,tnd->dt', c, B).astype(np.float32)
    return y


def bench(fn):
    fn()  # compile + warm
    best = 1e18
    for _ in range(5):
        t0 = time.perf_counter(); r = fn(); best = min(best, (time.perf_counter() - t0) * 1e3)
    return best, r


t1, y1 = bench(lambda: scan(a, b, c, x))
t2, y2 = bench(lambda: parallel_scan(a, b, c, x))
rel1 = float(np.max(np.abs(y1 - haly)) / (np.max(np.abs(haly)) + 1e-6))
rel2 = float(np.max(np.abs(y2 - haly)) / (np.max(np.abs(haly)) + 1e-6))
print(f"Mamba selective SSM  N={N} D={D} T={T}")
print(f"  Halide inductive (sequential fold):   {hal_ms:8.3f} ms")
print(f"  Numba sequential per-channel scan:    {t1:8.3f} ms  (rel {rel1:.1e})")
print(f"  Assoc. PARALLEL scan (SOTA algo):     {t2:8.3f} ms  (rel {rel2:.1e})")
