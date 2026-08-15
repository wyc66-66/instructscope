"""Custom multi-object pick environment built on robosuite's Lift task.

The stock Lift task has a single cube; we need several colour-distinct cubes
on the table so that an *instruction* (not a fixed routine) is what selects
the target. We subclass Lift and swap the single cube for ``n_objects``
solid-colour boxes, each with its own body id. Everything else (Panda arm,
OSC controller, agentview camera) is inherited.

The environment is deterministic per ``seed``: object positions are sampled
on a fixed grid with per-object jitter, so scene-to-scene variation is
controlled and reproducible.
"""
from __future__ import annotations

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

# A fixed colour palette; instruction construction keys off these names.
PALETTE = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
    "green": [0.0, 0.6, 0.0, 1.0],
    "yellow": [1.0, 0.8, 0.0, 1.0],
    "purple": [0.6, 0.0, 0.8, 1.0],
    "orange": [1.0, 0.5, 0.0, 1.0],
}


class MultiLift(Lift):
    """Lift task with several colour-distinct cubes instead of one.

    ``obj_colors`` selects a subset of :data:`PALETTE`. The agentview camera
    sees all objects; the task is to lift the object named by an instruction.
    """

    def __init__(
        self,
        robots,
        obj_colors=("red", "blue", "green", "yellow"),
        target=None,
        seed=0,
        **kwargs,
    ):
        self.obj_colors = list(obj_colors)
        self.target = target if target is not None else self.obj_colors[0]
        self._rng = np.random.RandomState(seed)
        self.cubes = {}
        self.cube_body_ids = {}
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        """Build the table + robot + N colour cubes model."""
        super()._load_model()

        from robosuite.models.arenas import TableArena

        # Build the table arena (same as Lift's, kept as an attribute so the
        # success check and placement can reference it).
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        self.mujoco_arena.set_origin([0, 0, 0])

        # Place cubes on a spread grid so colours are clearly separated in
        # the agentview image. Positions are in the table frame.
        n = len(self.obj_colors)
        grid_pos = _grid_positions(n, self._rng)

        self.cubes = {}
        self.placement_initializer = None
        for i, color in enumerate(self.obj_colors):
            rgba = list(PALETTE[color])
            # Slightly muted so the VLM doesn't overfit to saturated colours
            rgba = [0.85 * c + 0.1 for c in rgba]
            rgba[3] = 1.0
            cube = BoxObject(
                name=f"cube_{color}",
                size_min=[0.022, 0.022, 0.022],
                size_max=[0.024, 0.024, 0.024],
                rgba=rgba,
            )
            self.cubes[color] = (cube, grid_pos[i])

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[cube for cube, _ in self.cubes.values()],
        )

    def _setup_references(self):
        # Lift's parent _setup_references expects self.cube to exist and
        # resolves its body id; give it the first coloured cube so the
        # inherited reward()/grasp checks keep working.
        self.cube = next(iter(self.cubes.values()))[0]
        super()._setup_references()
        self.cube_body_ids = {
            color: self.sim.model.body_name2id(cube.root_body)
            for color, (cube, _) in self.cubes.items()
        }
        self.cube_body_id = self.cube_body_ids[self.obj_colors[0]]

    def _setup_observables(self):
        """Add object observables for every cube (position, quaternion)."""
        observables = super()._setup_observables()

        if self.use_object_obs:
            for color, body_id in self.cube_body_ids.items():
                name = f"cube_{color}_pos"

                @sensor(modality="object")
                def cube_pos(obs_cache, body_id=body_id):
                    return np.array(self.sim.data.body_xpos[body_id])

                @sensor(modality="object")
                def cube_quat(obs_cache, body_id=body_id):
                    from robosuite.utils.transform_utils import convert_quat

                    return convert_quat(
                        np.array(self.sim.data.body_xquat[body_id]), to="xyzw"
                    )

                observables[name] = Observable(
                    name=name,
                    sensor=cube_pos,
                    sampling_rate=self.control_freq,
                )
                observables[f"{name}_quat"] = Observable(
                    name=f"{name}_quat",
                    sensor=cube_quat,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _reset_internal(self):
        """Place each cube at its fixed grid position (deterministic per seed).

        We deliberately call ``SingleArmEnv._reset_internal`` (not Lift's) so
        that the single-cube placement sampler in Lift is never touched, then
        write each cube's qpos like Lift's sampler would.
        """
        from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

        SingleArmEnv._reset_internal(self)
        for color, (cube, pos) in self.cubes.items():
            x, y = pos
            qpos = np.array([x, y, self.table_offset[2] + 0.025, 1.0, 0.0, 0.0, 0.0])
            self.sim.data.set_joint_qpos(cube.joints[0], qpos)
        self.sim.forward()

    def _check_success(self):
        """True if the target cube is lifted above the table by a margin."""
        target_body = self.cube_body_ids[self.target]
        cube_height = self.sim.data.body_xpos[target_body][2]
        table_height = self.table_offset[2]
        return cube_height > table_height + 0.04

    def reward(self, action=None):
        if self._check_success():
            return 2.25 if self.reward_scale is None else 2.25 * self.reward_scale
        return 0.0


def _grid_positions(n: int, rng: np.random.RandomState):
    """Deterministic spread of n cube positions in the table frame."""
    cols = max(2, int(np.ceil(np.sqrt(n))))
    positions = []
    xs = np.linspace(-0.2, 0.2, cols)
    for i in range(n):
        row = i // cols
        col = i % cols
        x = xs[col] + rng.uniform(-0.01, 0.01)
        y = 0.20 - 0.20 * row + rng.uniform(-0.01, 0.01)
        positions.append((x, y))
    return positions
