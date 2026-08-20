"""Third-party reference for the batched steady-state Kalman recurrence, using
NumPy + BLAS (the standard optimized way): per timestep a batched matvec
X = A @ X + K z_t (N x B, BLAS-threaded), y_t = C @ X. Validates + times against
the Halide inductive result."""
import time
import numpy as np

with open("apps/kalman_ss/params.txt") as f:
    N, B, T, hal_ms = f.readline().split()
N, B, T, hal_ms = int(N), int(B), int(T), float(hal_ms)

A = np.fromfile("apps/kalman_ss/A.bin", np.float32).reshape(N, N)  # A(j,i)=Ap[j+i*N] => row i
K = np.fromfile("apps/kalman_ss/K.bin", np.float32)
C = np.fromfile("apps/kalman_ss/C.bin", np.float32)
z = np.fromfile("apps/kalman_ss/z.bin", np.float32).reshape(T, B).T  # Halide z(b,t)=zp[b+t*B]
haly = np.fromfile("apps/kalman_ss/y.bin", np.float32).reshape(B, T)

# C++ layout A(j,i)=Ap[j+i*N]: element (row=i, col=j). x_t(i)=sum_j A(j,i) x_{t-1}(j)
# => x = Arow @ x where Arow[i,j] = A(j,i) = A.reshape(N,N)[i,j]. So A as read (i,j) is correct.
def run():
    y = np.empty((B, T), np.float32)
    X = K[:, None] * z[:, 0][None, :]          # N x B
    y[:, 0] = C @ X
    for t in range(1, T):
        X = A @ X + K[:, None] * z[:, t][None, :]
        y[:, t] = C @ X
    return y

y = run()  # warm up
best = 1e18
for _ in range(5):
    t0 = time.perf_counter(); y = run(); best = min(best, (time.perf_counter() - t0) * 1e3)

rel = float(np.max(np.abs(y - haly)) / (np.max(np.abs(haly)) + 1e-6))
print(f"Steady-state Kalman  N={N} B={B} T={T}")
print(f"  Halide inductive:            {hal_ms:8.3f} ms")
print(f"  NumPy + BLAS (third-party):  {best:8.3f} ms")
print(f"  max rel error (Halide vs NumPy) = {rel:.3e} -> {'PASS' if rel < 1e-3 else 'FAIL'}")
