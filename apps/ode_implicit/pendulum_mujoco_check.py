"""Ground-truth + third-party comparison for pendulum_mujoco.cpp: runs the
SAME damped hinge pendulum in actual MuJoCo (fixed step, integrator=
"implicit" -- exactly the real fixed-step physics-engine case discussed:
MuJoCo/Bullet/PhysX use fixed step + fixed structure for real-time/robotics
simulation, same reason dSPACE/OPAL-RT HIL platforms and Simulink's ode14x
use fixed step for real-time control).

This file also documents the derivation used in pendulum_mujoco.cpp:
  qfrc_bias(theta) = -A*cos(theta), A=2.4525 (fit to MuJoCo's own
    qfrc_bias output at several theta values, residual ~1e-16)
  M_eff = M + h*damping  (MuJoCo's implicit-integrator mass augmentation
    for the damping term, verified against MuJoCo's actual qvel/qpos after
    one mj_step to match a hand-computed value exactly)
"""
import time
import numpy as np
import mujoco

with open("apps/ode_implicit/pend_params.txt") as f:
    B, T, hal_ms = f.readline().split()
B, T, hal_ms = int(B), int(T), float(hal_ms)
theta0 = np.fromfile("apps/ode_implicit/pend_theta0.bin", np.float64)
obs = np.fromfile("apps/ode_implicit/pend_obs.bin", np.float64).reshape(B, T)

XML = """
<mujoco>
  <option timestep="0.001" integrator="implicit" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pendulum" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="2.0"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.02" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(XML)

N_SAMPLE = min(64, B)
diffs = []
t0 = time.perf_counter()
d = mujoco.MjData(m)  # reused across trajectories, like the CVODE fix
for b in range(N_SAMPLE):
    mujoco.mj_resetData(m, d)
    d.qpos[0] = theta0[b]
    d.qvel[0] = 0.0
    traj = np.empty(T)
    traj[0] = d.qpos[0]
    for t in range(1, T):
        mujoco.mj_step(m, d)
        traj[t] = d.qpos[0]
    diffs.append(float(np.max(np.abs(traj - obs[b]))))
elapsed_sample = (time.perf_counter() - t0) * 1e3
extrapolated_full = elapsed_sample * (B / N_SAMPLE)

print(f"MuJoCo-matching damped pendulum  B={B} T={T} (fixed step h=0.001, integrator=implicit)")
print(f"  Halide inductive:                          {hal_ms:8.3f} ms")
print(f"  MuJoCo (compiled C, same fixed step + same")
print(f"    implicit-mass-augmentation formula), {N_SAMPLE} trajectories")
print(f"    (Python mj_step call per step, no native batching): {elapsed_sample:8.3f} ms")
print(f"    -> extrapolated to full B={B}:            {extrapolated_full:8.3f} ms"
      f"  ({extrapolated_full / hal_ms:.0f}x slower than Halide)")
print(f"  max abs diff vs Halide's fixed-step trajectory: {max(diffs):.3g}"
      f"  (same algorithm -- this should be at the float64 noise floor)")
