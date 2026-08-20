#!/usr/bin/env python3
# FAIR compiled+batched third-party baseline for the kalman_ss workload:
# JAX with a JIT-compiled lax.scan over time and vmap over the batch (same
# strategy as kalman_ll_llt_jax.py). CPU backend, so it's an apples-to-apples
# compiled-CPU-vs-compiled-CPU comparison against Halide.
import numpy as np, time, jax, jax.numpy as jnp
from jax import lax, vmap, jit

with open("apps/kalman_ss/params.txt") as f:
    N, B, T, hal_ms = f.readline().split()
N, B, T, hal_ms = int(N), int(B), int(T), float(hal_ms)

jax.config.update("jax_enable_x64", False)

A = jnp.array(np.fromfile("apps/kalman_ss/A.bin", np.float32).reshape(N, N))  # A[i, j]
K = jnp.array(np.fromfile("apps/kalman_ss/K.bin", np.float32))
C = jnp.array(np.fromfile("apps/kalman_ss/C.bin", np.float32))
z_np = np.fromfile("apps/kalman_ss/z.bin", np.float32).reshape(T, B).T  # (B, T)
z = jnp.array(z_np)
haly = np.fromfile("apps/kalman_ss/y.bin", np.float32).reshape(B, T)


def scan_one_series(z_series):
    x0 = K * z_series[0]
    y0 = C @ x0

    def step(xp, zt):
        xn = A @ xp + K * zt
        yt = C @ xn
        return xn, yt

    _, ys = lax.scan(step, x0, z_series[1:])
    return jnp.concatenate([y0[None], ys])


batched = jit(vmap(scan_one_series))


def bench(fn):
    r = fn()
    r.block_until_ready()  # compile + warm
    best = 1e18
    for _ in range(5):
        t0 = time.perf_counter()
        r = fn()
        r.block_until_ready()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best, r


t_jax, y_jax = bench(lambda: batched(z))
y_jax = np.asarray(y_jax)

rel = float(np.max(np.abs(y_jax - haly)) / (np.max(np.abs(haly)) + 1e-6))
print(f"Steady-state Kalman  N={N} B={B} T={T}")
print(f"  Halide inductive:              {hal_ms:8.3f} ms")
print(f"  JAX lax.scan+vmap, jit (CPU):  {t_jax:8.3f} ms  (rel {rel:.1e})")
print(f"  ratio: JAX/inductive {t_jax / hal_ms:.1f}x")
