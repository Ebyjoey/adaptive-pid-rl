"""MuJoCo-backed inverted pendulum plant.

This class is the *one* physics implementation used by both the training
Gymnasium environment (``envs.gym_env``) and the ROS2 ``plant_node`` -- per
docs/architecture.md Section 1, duplicating physics between sim and ROS2 is
exactly the kind of divergence this project is designed to avoid.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from adaptive_pid.utils.types import PlantParams, PlantState

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "sim" / "mujoco_models" / "inverted_pendulum.xml"


class InvertedPendulumPlant:
    """Wraps a MuJoCo model of a torque-actuated inverted pendulum.

    Physical parameters (mass, damping, actuator gain, etc.) are applied by
    mutating the compiled ``MjModel`` in place (rather than recompiling XML
    per episode), which is both far faster and is the mechanism
    ``DomainRandomizer`` uses to implement payload variation, friction
    changes, and actuator degradation.
    """

    def __init__(self, model_path: str | Path | None = None, dt: float = 0.01) -> None:
        path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo model not found at {path}")

        self._model = mujoco.MjModel.from_xml_path(str(path))
        self._model.opt.timestep = dt
        self._data = mujoco.MjData(self._model)
        self._dt = dt

        # Cached indices, resolved once, to avoid repeated name->id lookups
        # on every step (mujoco name lookups are not free).
        self._pole_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "pole")
        self._joint_dof_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "pivot")
        self._actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pivot_motor")

        # actuator_gain is applied as a software multiplier on commanded
        # torque (representing actuator degradation) rather than a MuJoCo
        # gear-ratio edit, so it can be varied continuously within an episode
        # without touching the compiled model's actuator definition.
        self._actuator_gain_multiplier: float = 1.0

        self._nominal_tip_mass = float(self._model.body_mass[self._pole_body_id])
        self._nominal_damping = float(self._model.dof_damping[self._joint_dof_id])
        self._nominal_length = 0.5  # matches the "tip_mass" geom pos in the XML

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    def apply_params(self, params: PlantParams) -> None:
        """Overwrite the compiled model's physical parameters in place.

        Called by ``DomainRandomizer`` at episode reset, and optionally
        mid-episode for the subset of parameters that vary over time
        (actuator degradation, battery-voltage-driven gain droop).
        """
        self._model.body_mass[self._pole_body_id] = params.mass
        self._model.dof_damping[self._joint_dof_id] = params.damping

        # Update the body's rotational inertia to stay physically consistent
        # with the new point mass at the existing lever arm length (thin-rod
        # + point-mass approximation): I = m * l^2 for the tip point mass,
        # plus a small extra inertia term representing unmodeled rotor
        # inertia / dynamics that domain randomization also perturbs.
        lever_arm = params.length
        point_mass_inertia = params.mass * lever_arm**2
        self._model.body_inertia[self._pole_body_id] = [
            point_mass_inertia + params.inertia_extra,
            point_mass_inertia + params.inertia_extra,
            1e-4,  # negligible inertia about the pendulum's own long axis
        ]

        # Reposition the tip geometry/site to reflect the new pendulum length
        # by scaling the body's single child geom's "fromto" isn't directly
        # mutable at runtime in MuJoCo without recompiling, so instead we
        # keep geometry fixed and represent length changes purely through
        # the inertia/mass relationship above -- physically this models a
        # payload of different mass at a fixed attachment point, which is
        # the intended "payload variation" scenario (docs/mdp_design.md).
        self._actuator_gain_multiplier = params.actuator_gain

    def reset(self, initial_theta: float = 0.0, initial_theta_dot: float = 0.0) -> PlantState:
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[0] = initial_theta
        self._data.qvel[0] = initial_theta_dot
        mujoco.mj_forward(self._model, self._data)
        return self.get_state()

    def step(self, control_torque: float, disturbance_torque: float = 0.0) -> PlantState:
        """Advance the simulation by one ``dt``.

        Parameters
        ----------
        control_torque:
            Commanded torque from the PID controller (N*m), before actuator
            degradation is applied.
        disturbance_torque:
            Exogenous disturbance torque (N*m) injected this step, e.g. from
            ``envs.domain_randomization`` or the ROS2 ``disturbance_node``.
        """
        effective_torque = control_torque * self._actuator_gain_multiplier
        self._data.ctrl[self._actuator_id] = effective_torque

        # Disturbance is applied as a direct generalized force on the hinge
        # dof, additive to the actuator torque -- this represents an
        # external torque (e.g. wind gust, bump, cable drag) rather than a
        # further actuator-side effect.
        self._data.qfrc_applied[self._joint_dof_id] = disturbance_torque

        mujoco.mj_step(self._model, self._data)
        return self.get_state()

    def get_state(self) -> PlantState:
        return PlantState(
            theta=float(self._data.qpos[0]),
            theta_dot=float(self._data.qvel[0]),
            time=float(self._data.time),
        )

    def get_noisy_observation(self, theta_noise_std: float, theta_dot_noise_std: float, rng: np.random.Generator) -> PlantState:
        """Return the plant state corrupted by additive Gaussian sensor
        noise, representing real encoder/gyro quantization and noise floors.
        The *true* internal state is unaffected -- only what an observer
        (PID controller, RL agent) is allowed to see is corrupted, which is
        the physically correct place to inject sensor noise.
        """
        true_state = self.get_state()
        return PlantState(
            theta=true_state.theta + float(rng.normal(0.0, theta_noise_std)),
            theta_dot=true_state.theta_dot + float(rng.normal(0.0, theta_dot_noise_std)),
            time=true_state.time,
        )
