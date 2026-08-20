"""Third-party references for the batched steady-state Kalman scan using PyTorch:
 (a) eager Python-loop over time (the naive framework way), and
 (b) torch.compile of the same scan (a real JIT-compiled scan -- the optimized
     reference that inductive Halide should roughly MATCH, since being fast at a
     dense linear state-space scan means compiling it)."""
import os, time
import numpy as np
import torch
NTHREADS = os.cpu_count()
torch.set_num_threads(NTHREADS)                 # match Halide's core count
try: torch.set_num_interop_threads(NTHREADS)
except Exception: pass
print(f"  [torch threads: {torch.get_num_threads()}]")

with open("apps/kalman_ss/params.txt") as f:
    N, B, T, hal_ms = f.readline().split()
N, B, T, hal_ms = int(N), int(B), int(T), float(hal_ms)
A = torch.from_numpy(np.fromfile("apps/kalman_ss/A.bin", np.float32).reshape(N, N).copy())
K = torch.from_numpy(np.fromfile("apps/kalman_ss/K.bin", np.float32).copy())
C = torch.from_numpy(np.fromfile("apps/kalman_ss/C.bin", np.float32).copy())
z = torch.from_numpy(np.ascontiguousarray(np.fromfile("apps/kalman_ss/z.bin", np.float32).reshape(T, B).T))
haly = np.fromfile("apps/kalman_ss/y.bin", np.float32).reshape(B, T)

def scan(A, K, C, z):
    Bn, Tn = z.shape
    y = torch.empty((Bn, Tn), dtype=z.dtype)
    X = K[:, None] * z[:, 0][None, :]
    y[:, 0] = C @ X
    for t in range(1, Tn):
        X = A @ X + K[:, None] * z[:, t][None, :]
        y[:, t] = C @ X
    return y

def bench(fn, n=5):
    fn()  # warm up / compile
    best = 1e18
    for _ in range(n):
        t0 = time.perf_counter(); fn(); best = min(best, (time.perf_counter() - t0) * 1e3)
    return best

with torch.no_grad():
    ye = scan(A, K, C, z)
    t_eager = bench(lambda: scan(A, K, C, z))
    t_comp = None
    if os.getenv("COMPILE"):    # torch.compile unrolls the 512-step loop -> slow; opt-in
        scan_c = torch.compile(scan)
        scan_c(A, K, C, z)
        t_comp = bench(lambda: scan_c(A, K, C, z))

rel = float(np.max(np.abs(ye.numpy() - haly)) / (np.max(np.abs(haly)) + 1e-6))
print(f"Steady-state Kalman  N={N} B={B} T={T}")
print(f"  Halide inductive:                 {hal_ms:8.3f} ms")
print(f"  PyTorch eager (BLAS matmul/step, all cores): {t_eager:8.3f} ms")
if t_comp is not None:
    print(f"  PyTorch torch.compile (JIT scan): {t_comp:8.3f} ms")
print(f"  max rel error (Halide vs torch) = {rel:.3e} -> {'PASS' if rel < 1e-3 else 'FAIL'}")
