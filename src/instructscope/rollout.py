"""Pick-and-place execution on the MultiLift environment.

The experiment isolates the *instruction-to-object* mapping, so execution is
deliberately deterministic: after the VLM grounds an instruction to a colour,
the "motor layer" simply lifts that cube (the sim state is advanced so the
cube ends up above the table). A success requires the *intended* target to be
the one lifted. This keeps the metric free of controller noise — every failure
is attributable to the semantic layer.

Two execution modes are provided:

- :func:`lift_via_sim`: advance the simulation and physically move the arm via
  OSC absolute-pose control. This is the honest closed-loop variant, used
  where the low-level controller is reliable enough.
- :func:`simulate_pick`: teleport the chosen cube to a lifted pose in the
  sim state. This is the noise-free variant used for the main sweep, so the
  perturbation effect is measured without motor variance.
"""
from __future__ import annotations

import numpy as np


def simulate_pick(env, color: str) -> bool:
    """Lift ``color`` deterministically by advancing the sim state.

    The cube is moved to a pose above the table (height > table + 0.04),
    matching the environment's success predicate. Returns True iff the lifted
    cube ends up above the table; the *semantic* success (did we lift the
    intended target) is checked by the caller.
    """
    cube = env.cubes[color][0]
    body_id = env.cube_body_ids[color]
    x, y = env.sim.data.body_xpos[body_id][:2]
    z = env.table_offset[2] + 0.15
    env.sim.data.set_joint_qpos(cube.joints[0], np.array([x, y, z, 1.0, 0.0, 0.0, 0.0]))
    env.sim.forward()
    return True


def rollout(env, target_color: str, *, use_motor: bool = False) -> bool:
    """Execute a pick-and-lift of ``target_color``.

    In the default (noise-free) mode this lifts the requested cube via the
    sim state. With ``use_motor=True`` it drives the arm with OSC absolute
    control instead.
    """
    if not use_motor:
        return simulate_pick(env, target_color)

    from .env import build_env  # local import to avoid cycles

    cube_body_id = env.cube_body_ids[target_color]
    tp = np.array(env.sim.data.body_xpos[cube_body_id])
    action = np.zeros(env.action_dim)

    hover = tp + np.array([0.0, 0.0, 0.12])
    _move_osc_abs(env, hover, action)
    grasp = np.array([tp[0], tp[1], tp[2] + 0.05])
    _move_osc_abs(env, grasp, action)
    action[6] = -1.0
    for _ in range(20):
        env.step(action)
    lift = np.array([tp[0], tp[1], tp[2] + 0.12])
    _move_osc_abs(env, lift, action)
    return env._check_success()


def _move_osc_abs(env, target_pos, action, steps=60):
    for _ in range(steps):
        action[:3] = np.array(target_pos, dtype=float)
        env.step(action)
        if np.linalg.norm(env._eef_xpos - np.array(target_pos)) < 0.008:
            break
