#!/usr/bin/env python3
# Full-trajectory Kalman FILTER baseline in JAX: lax.scan carrying (x,P), but
# only STACKING the states x. Because P lives in the scan carry (not the stacked
# output), XLA keeps it as O(d^2) loop state -- i.e. lax.scan folds P for free,
# structurally like the inductive Halide fold. We also probe peak memory to
# check P is not materialized as an O(B*T*d^2) array.
import numpy as np, time, tracemalloc, jax, jax.numpy as jnp
from jax import lax, vmap, jit

with open("traj_params.txt") as f:
    B, T, tind, tmat, q0, q1, Rv = f.read().split()
B, T = int(B), int(T); tind, tmat, q0, q1, Rv = map(float, (tind, tmat, q0, q1, Rv))

z = np.fromfile("traj_z.bin", dtype=np.float64).reshape(T, B).T          # (B,T)
lvl_h = np.fromfile("traj_level.bin", dtype=np.float64).reshape(T, B).T   # (B,T)
trd_h = np.fromfile("traj_trend.bin", dtype=np.float64).reshape(T, B).T

jax.config.update("jax_enable_x64", True)
F = jnp.array([[1.0, 1.0], [0.0, 1.0]]); H = jnp.array([[1.0, 0.0]])
Q = jnp.array([[q0, 0.0], [0.0, q1]]); R = jnp.array([[Rv]])

def series_filter(zs):                 # zs: (T-1,) observations z[1..T-1]
    def step(carry, zt):
        x, P = carry                   # x:(2,1) P:(2,2)  <- carried, NOT stacked
        Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ jnp.linalg.inv(S)
        xp = F @ x
        x = xp + K * (zt - (H @ xp)[0, 0])
        P = (jnp.eye(2) - K @ H) @ Pp
        return (x, P), x[:, 0]          # STACK only the state x (2,)
    x0 = jnp.zeros((2, 1)); P0 = jnp.eye(2)
    _, xs = lax.scan(step, (x0, P0), zs)
    return xs                           # (T-1, 2)

batched = jit(vmap(series_filter))
zj = jnp.asarray(z[:, 1:])
xs = np.asarray(batched(zj).block_until_ready())        # (B, T-1, 2)

best = 1e18
tracemalloc.start()
for _ in range(10):
    t0 = time.perf_counter(); batched(zj).block_until_ready()
    best = min(best, (time.perf_counter() - t0) * 1e3)
_, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

# validate states vs Halide (Halide stores t=0..T-1; JAX produced t=1..T-1)
lvl_j = xs[:, :, 0]; trd_j = xs[:, :, 1]
el = np.max(np.abs(lvl_j - lvl_h[:, 1:])); et = np.max(np.abs(trd_j - trd_h[:, 1:]))
Pfull = B * T * 3 * 8 / 1048576.0
print(f"Kalman full-trajectory filter (d=2)  B={B} T={T}")
print(f"  Halide inductive (fold P->2, x full):  {tind:8.3f} ms")
print(f"  Halide materialize (P full + x full):  {tmat:8.3f} ms")
print(f"  JAX lax.scan (P in carry, stack x):    {best:8.3f} ms")
print(f"  state err vs Halide: level {el:.3g} trend {et:.3g} -> {'PASS' if max(el,et)<1e-6 else 'FAIL'}")
print(f"  (P history if materialized would be {Pfull:.0f} MB;")
print(f"   JAX python peak alloc during timed runs: {peak/1048576.0:.1f} MB -> P {'NOT materialized' if peak/1048576.0 < Pfull*0.5 else 'materialized'})")
