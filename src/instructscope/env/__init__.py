"""robosuite environment construction helpers."""
from __future__ import annotations

from .multilift import MultiLift, PALETTE

__all__ = ["MultiLift", "PALETTE"]


def build_env(obj_colors, target, seed=0, camera="agentview", img_size=256,
              use_camera_obs=True):
    """Create a MultiLift env, returning (env, target_color).

    ``img_size`` is used for both the offscreen render buffer and the
    observation camera so the EGL context and the obs buffer always match
    (a mismatch silently yields black frames on some WSL/EGL setups).
    ``use_camera_obs`` can be disabled for pure-control rollouts to avoid
    re-rendering on every step (rendering is only needed for the grounding
    snapshot, which is taken once per scene).

    The robot uses an OSC_POSE controller in absolute-pose mode
    (``control_delta=False``) so the scripted rollout can drive the eef to
    world coordinates directly.
    """
    from robosuite.controllers import load_controller_config

    osc = load_controller_config(default_controller="OSC_POSE")
    osc["control_delta"] = False

    env = MultiLift(
        robots="Panda",
        obj_colors=obj_colors,
        target=target,
        seed=seed,
        controller_configs=osc,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera=camera,
        camera_heights=img_size,
        camera_widths=img_size,
        control_freq=20,
        use_object_obs=True,
        use_camera_obs=use_camera_obs,
    )
    env.reset()
    return env
