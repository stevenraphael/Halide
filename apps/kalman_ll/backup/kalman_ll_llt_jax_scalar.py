#!/usr/bin/env python3
# FAIR compiled+batched third-party baseline for the kalman_ll_llt workload:
# JAX with a JIT-compiled lax.scan over time and vmap over the batch. This is
# what dynamax / TFP do internally (XLA-compiled scan + batching), without the
# per-timestep Python loop that makes simdkalman slow. CPU backend here, so it's
# an apples-to-apples compiled-CPU-vs-compiled-CPU comparison against Halide.
import numpy as np, time, jax, jax.numpy as jnp
from jax import lax, vmap, jit

with open("params_llt.txt") as f:
    B, T, ti, tn, q0, q1, Rv = f.read().split()
B, T = int(B), int(T)
ti, tn, q0, q1, Rv = map(float, (ti, tn, q0, q1, Rv))

z = np.fromfile("z_llt.bin", dtype=np.float64).reshape(T, B).T   # (B, T)
ll_halide = np.fromfile("ll_llt.bin", dtype=np.float64)          # (B,)

jax.config.update("jax_enable_x64", True)

F = jnp.array([[1.0, 1.0], [0.0, 1.0]])
H = jnp.array([[1.0, 0.0]])
Q = jnp.array([[q0, 0.0], [0.0, q1]])
R = jnp.array([[Rv]])
x0 = jnp.zeros((2, 1))
P0 = jnp.eye(2)

def series_loglik(zs):
    # zs: (T-1,) observations z[1..T-1], matching Halide's summation range.
    def step(carry, zt):
        x, P, ll = carry
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R                    # (1,1)
        K = (Pp @ H.T) / S[0, 0]           # (2,1) -- S is 1x1, scalar divide instead of matrix inverse
        xp = F @ x
        innov = zt - (H @ xp)[0, 0]
        ll = ll + (-0.5) * (innov * innov / S[0, 0] + jnp.log(S[0, 0]))
        x = xp + K * innov
        P = (jnp.eye(2) - K @ H) @ Pp
        return (x, P, ll), None
    (_, _, ll), _ = lax.scan(step, (x0, P0, 0.0), zs)
    return ll

batched = jit(vmap(series_loglik))

zj = jnp.asarray(z[:, 1:])                      # (B, T-1)
ll_jax = np.asarray(batched(zj).block_until_ready())   # compile + run

best = 1e18
for _ in range(5):
    t0 = time.perf_counter()
    batched(zj).block_until_ready()
    best = min(best, (time.perf_counter() - t0) * 1e3)

abs_err = np.max(np.abs(ll_jax - ll_halide))
rel_err = np.max(np.abs(ll_jax - ll_halide) / np.maximum(np.abs(ll_halide), 1.0))
print(f"Kalman local-linear-trend log-likelihood  B={B} T={T}  (|LL| ~ {np.median(np.abs(ll_halide)):.0f})")
print(f"  Halide inductive (fold):        {ti:8.3f} ms")
print(f"  Halide non-inductive (mat.):    {tn:8.3f} ms")
print(f"  JAX lax.scan+vmap, jit (CPU):   {best:8.3f} ms")
print(f"  err vs Halide LL: abs {abs_err:.3g}  rel {rel_err:.3g}  -> {'PASS' if rel_err < 1e-4 else 'FAIL'}")
print(f"  ratio: JAX/inductive {best/ti:.1f}x, JAX/non-inductive {best/tn:.1f}x")
