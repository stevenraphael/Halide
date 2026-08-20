"""Optimized, natively-batched third-party comparison for pendulum_mujoco.cpp:
MJX (mujoco.mjx), Google's JAX-based reimplementation of MuJoCo, vmap'd +
jit'd across trajectories -- this is the actual "designed for many parallel
physics environments" optimized library, unlike calling mj_step() from a
Python loop (which mostly measures interpreter overhead, not MuJoCo's real
per-step cost -- see pendulum_mujoco_check.py's 2500x number and the
follow-up discussion).

Uses integrator="implicitfast" (MJX doesn't support "implicit"); for this
single-DOF system with no Coriolis term the two are identical (verified:
mj_step with "implicit" and mjx.step with "implicitfast" agree to float32
precision on the same model).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

with open("apps/ode_implicit/pend_params.txt") as f:
    B, T, hal_ms = f.readline().split()
B, T, hal_ms = int(B), int(T), float(hal_ms)
theta0 = np.fromfile("apps/ode_implicit/pend_theta0.bin", np.float64)
obs = np.fromfile("apps/ode_implicit/pend_obs.bin", np.float64).reshape(B, T)

XML = """
<mujoco>
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="2.0"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.02" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)
mx = mjx.put_model(m)
d0 = mjx.put_data(m, mujoco.MjData(m))


def rollout(theta_init):
    dx = d0.replace(qpos=jnp.array([theta_init]), qvel=jnp.array([0.0]))

    def body(dx, _):
        dx = mjx.step(mx, dx)
        return dx, dx.qpos[0]

    _, traj = jax.lax.scan(body, dx, None, length=T - 1)
    return jnp.concatenate([jnp.array([theta_init]), traj])


batched_rollout = jax.jit(jax.vmap(rollout))

N_SAMPLE = min(256, B)
thetas = jnp.array(theta0[:N_SAMPLE])
traj = batched_rollout(thetas)  # warm up / compile
traj.block_until_ready()
t0 = time.perf_counter()
traj = batched_rollout(thetas)
traj.block_until_ready()
elapsed_sample = (time.perf_counter() - t0) * 1e3
extrapolated_full = elapsed_sample * (B / N_SAMPLE)

traj_np = np.array(traj)
diff = float(np.max(np.abs(traj_np - obs[:N_SAMPLE])))

print(f"MuJoCo-matching damped pendulum  B={B} T={T} (fixed step h=0.001)")
print(f"  Halide inductive:                                {hal_ms:8.3f} ms")
print(f"  MJX (JAX-based MuJoCo, vmap+jit batched), {N_SAMPLE} trajectories:")
print(f"                                                    {elapsed_sample:8.3f} ms")
print(f"    -> extrapolated to full B={B}:                  {extrapolated_full:8.3f} ms"
      f"  ({extrapolated_full / hal_ms:.1f}x vs Halide)")
print(f"  max abs diff vs Halide's fixed-step trajectory:   {diff:.3g}"
      f"  (implicitfast vs implicit -- identical here, no Coriolis term to differ)")
