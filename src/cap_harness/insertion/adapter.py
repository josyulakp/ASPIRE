"""Frame-explicit adapter for the PR V4P1SR + chopsticks insertion scenes.

The robot starts holding the plug ``standoff`` above the socket mouth (keyframe
``initial`` of an insertion_skill scene). Generated programs see only what a real
station would give them without a force/torque sensor:

* proprioception -- joint positions/velocities and the *nominal* plug-tip pose, i.e.
  forward kinematics composed with the grasp the planner intended;
* a perception-grade estimate of the seated plug-tip pose (and the socket mouth);
* RGB-D from a close-up and an overview camera.

Each seed perturbs the episode with two errors the program cannot read directly: the
plug sits in the jaws slightly off the nominal grasp (the weld is moved), and the
socket estimate is off the true socket. Seed 0 is the unperturbed scene.

The arm runs a joint impedance controller (gravity compensation plus joint PD) at the
simulator rate; one control period of ``step`` holds a joint-position target for
``1 / control_frequency`` seconds. The low joint stiffness is deliberate: contact
shows up as tracking error rather than as a force reading.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    IKResult,
    Observation,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import invert_transform, matrix_to_pose
from cap_harness.insertion.registry import InsertionTaskMetadata, InsertionTaskRegistry

EMBODIMENT = "insertion"
BASE_FRAME = "pr_base_link"
EE_BODY = "pr_right_chop_base"

TRAJECTORY_TOLERANCE_RAD = 0.0005
TRAJECTORY_START_TOLERANCE_RAD = 0.02
TRAJECTORY_SETTLE_TICKS = 20
MAX_JOINT_SPEED_RAD_S = 0.5
SETTLE_AFTER_RESET_S = 0.1

#: Joint impedance gains. The shoulder/elbow gains are the ones insertion_skill.validate_envs
#: holds the arm with. Wrist joint 6 is 4x stiffer than there, so that a plug pressed lightly
#: onto a socket face can be slid across it instead of the wrist pivoting around the stuck
#: tip. Wrist roll (joint 7) keeps the soft gain: its axis is the insertion axis for every
#: axial grasp, so it gives passive yaw compliance, letting chamfers and guides turn the plug.
KP_JOINT = np.array([400.0, 400.0, 300.0, 200.0, 120.0, 30.0])
KD_JOINT = np.array([40.0, 40.0, 30.0, 20.0, 4.0, 2.0])

#: Per-seed perturbation bounds at scale 1 (uniform; lateral offsets uniform in a disc).
#: In-hand error is relative to the nominal plug-tip frame; socket-estimate error is
#: relative to the true seated-tip (insert_target) frame. Both frames have +z pointing
#: out of the socket.
IN_HAND_LATERAL_M = 0.001
IN_HAND_AXIAL_M = 0.0005
IN_HAND_TILT_RAD = math.radians(1.0)
IN_HAND_YAW_RAD = math.radians(1.0)
SOCKET_LATERAL_M = 0.001
SOCKET_AXIAL_M = 0.0005
SOCKET_TILT_RAD = math.radians(0.5)
SOCKET_YAW_RAD = math.radians(1.0)
#: Where the arm starts, relative to the true standoff pose (nominal tip frame): a lateral
#: offset, extra height, and a small misorientation. If the plug would then touch the
#: socket (tall guides, as on the DIMM slot), the start backs off along the axis.
START_LATERAL_M = 0.002
START_EXTRA_HEIGHT_M = 0.003
START_TILT_RAD = math.radians(1.0)
START_YAW_RAD = math.radians(1.0)
START_BACKOFF_STEP_M = 0.002
START_BACKOFF_MAX_M = 0.04
START_REDRAWS = 20
#: Opt-in overrides of the start-pose range, for testing robustness to where the arm starts:
#: INSERTION_START_LATERAL_M, INSERTION_START_HEIGHT_M (max extra height), INSERTION_START_TILT_DEG,
#: INSERTION_START_YAW_DEG. Unset, the defaults above apply.


def _transform(rotation: Any, position: Any) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return result


def _quat_to_matrix(quaternion_wxyz: Any) -> np.ndarray:
    import mujoco

    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion_wxyz, dtype=np.float64))
    return matrix.reshape(3, 3)


def _matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    import mujoco

    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation, dtype=np.float64).reshape(9))
    return quaternion


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    quaternion = _matrix_to_quat(rotation)
    if quaternion[0] < 0:
        quaternion = -quaternion
    sin_half = float(np.linalg.norm(quaternion[1:]))
    if sin_half < 1e-12:
        return np.zeros(3)
    return quaternion[1:] / sin_half * 2.0 * math.atan2(sin_half, quaternion[0])


def _random_offset(
    rng: np.random.Generator, lateral: float, axial: float, tilt: float, yaw: float
) -> np.ndarray:
    """A 4x4 offset: translation in a lateral disc plus axial, rotation tilt (x, y) and yaw (z)."""
    radius = lateral * math.sqrt(rng.uniform())
    angle = rng.uniform(0.0, 2.0 * math.pi)
    translation = np.array(
        [radius * math.cos(angle), radius * math.sin(angle), rng.uniform(-axial, axial)]
    )
    rotvec = np.array(
        [rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), rng.uniform(-yaw, yaw)]
    )
    return _transform(_rotvec_to_matrix(rotvec), translation)


def _pose_record(pose: Pose) -> dict[str, object]:
    return {
        "position": [float(value) for value in pose.position],
        "quaternion_wxyz": [float(value) for value in pose.quaternion_wxyz],
        "frame": pose.frame,
    }


FTS_BODY = "pr_right_fts"
FTS_SITE = "pr_fts"
#: Samples the tare averages (0.1 s at the 2 kHz simulator rate), as a real tare would.
FT_TARE_SAMPLES = 200
#: Torque noise and drift, per newton of the force knobs: about an ATI Mini45's ratio.
FT_TORQUE_PER_FORCE_M = 0.02


class _WristForceSensor:
    """Opt-in (INSERTION_FORCE=1) wrist F/T channel: the contact wrench on the plug.

    The MuJoCo sensor at ``pr_fts`` reads the load the arm carries outboard of the wrist.
    Each simulator substep this subtracts the predicted gravity load of the hand and the
    held plug (a real station's payload compensation, so reorienting the wrist does not
    look like contact), negates the rest so it is the wrench the *environment* applies to
    the plug, and filters it. ``reset`` tares the result after the episode has settled.
    The reported wrench is moved to the nominal plug tip and expressed in the nominal tip
    frame: +z points out of the socket, so pressing the plug into a face reads +Fz.

    Realism knobs, all off by default: INSERTION_FT_NOISE_N (white noise std per sample, N),
    INSERTION_FT_BIAS_N (bias drift, N per minute, in a random direction per episode; the
    tare removes only what has drifted by reset), INSERTION_FT_LOWPASS_HZ (first-order
    low-pass cutoff; 0 averages each control period instead). Torque noise and drift are
    ``FT_TORQUE_PER_FORCE_M`` times the force ones.
    """

    def __init__(self) -> None:
        self.noise_n = float(os.environ.get("INSERTION_FT_NOISE_N", "0"))
        self.bias_n_per_min = float(os.environ.get("INSERTION_FT_BIAS_N", "0"))
        self.lowpass_hz = float(os.environ.get("INSERTION_FT_LOWPASS_HZ", "0"))
        if min(self.noise_n, self.bias_n_per_min, self.lowpass_hz) < 0:
            raise ValueError("INSERTION_FT_NOISE_N, _BIAS_N and _LOWPASS_HZ must be non-negative")
        self._scale = np.r_[np.ones(3), np.full(3, FT_TORQUE_PER_FORCE_M)]
        self._rng = np.random.default_rng()
        self._drift_rate = np.zeros(6)
        self._tare = np.zeros(6)
        self._filtered = np.zeros(6)
        self._period_sum = np.zeros(6)
        self._period_count = 0
        self._wrench = np.zeros(6)

    def reset(self, adapter: InsertionAdapter, seed: int) -> None:
        """Bind to the new episode and tare; call once the reset has settled."""
        env = adapter.native_env
        model = env.model
        fts = model.body(FTS_BODY).id
        self._site = model.site(FTS_SITE).id
        outboard = [b for b in range(model.nbody) if b == fts or self._descends(model, b, fts)]
        self._bodies = np.array(outboard + [model.body("plug").id])
        self._masses = model.body_mass[self._bodies]
        self._env = env
        self._adapter = adapter
        self._t0 = float(env.data.time)
        # A separate stream, so enabling force leaves every seeded perturbation unchanged.
        self._rng = np.random.default_rng([seed, 0xF7])
        direction = self._rng.normal(size=6)
        direction[:3] /= np.linalg.norm(direction[:3])
        direction[3:] /= np.linalg.norm(direction[3:])
        self._drift_rate = direction * self._scale * self.bias_n_per_min / 60.0
        self._tare = np.zeros(6)
        self._tare = np.mean([self._contact_sample() for _ in range(FT_TARE_SAMPLES)], axis=0)
        self._filtered = np.zeros(6)
        self._period_sum[:] = 0.0
        self._period_count = 0
        self.end_period()

    @staticmethod
    def _descends(model: Any, body: int, ancestor: int) -> bool:
        while body > 0:
            body = int(model.body_parentid[body])
            if body == ancestor:
                return True
        return False

    def _contact_sample(self) -> np.ndarray:
        """One sensor-frame sample of the external wrench at the sensor, minus the tare."""
        data, model = self._env.data, self._env.model
        rotation = data.site_xmat[self._site].reshape(3, 3)
        origin = data.site_xpos[self._site]
        weight = self._masses[:, None] * model.opt.gravity[None, :]
        gravity_force = weight.sum(axis=0)
        gravity_torque = np.cross(data.xipos[self._bodies] - origin, weight).sum(axis=0)
        raw = self._env.wrist_wrench()
        raw = raw + self._drift_rate * (float(data.time) - self._t0)
        if self.noise_n > 0:
            raw = raw + self._rng.normal(scale=self.noise_n, size=6) * self._scale
        # The sensor carries the payload (raw = -gravity - contact), so contact = -(raw + gravity).
        contact = -(raw + np.r_[rotation.T @ gravity_force, rotation.T @ gravity_torque])
        return contact - self._tare

    def sample(self) -> None:
        """Advance the filter by one simulator substep."""
        value = self._contact_sample()
        if self.lowpass_hz > 0:
            alpha = 1.0 - math.exp(-2.0 * math.pi * self.lowpass_hz * self._env.model.opt.timestep)
            self._filtered += alpha * (value - self._filtered)
        self._period_sum += value
        self._period_count += 1

    def end_period(self) -> None:
        """Latch the reading for this control period."""
        if self.lowpass_hz > 0:
            sensor = self._filtered.copy()
        elif self._period_count:
            sensor = self._period_sum / self._period_count
        else:
            sensor = self._contact_sample()
        self._period_sum[:] = 0.0
        self._period_count = 0
        self._wrench = self._to_tip(sensor)

    def _to_tip(self, sensor: np.ndarray) -> np.ndarray:
        data = self._env.data
        world_from_sensor = data.site_xmat[self._site].reshape(3, 3)
        world_from_ee = _transform(data.xmat[self._adapter._ee_id], data.xpos[self._adapter._ee_id])
        world_from_tip = world_from_ee @ self._adapter._ee_from_tip_nominal
        force = world_from_sensor @ sensor[:3]
        torque = world_from_sensor @ sensor[3:]
        torque = torque + np.cross(data.site_xpos[self._site] - world_from_tip[:3, 3], force)
        tip_from_world = world_from_tip[:3, :3].T
        return np.r_[tip_from_world @ force, tip_from_world @ torque]

    @property
    def wrench(self) -> np.ndarray:
        return self._wrench.copy()

    def metadata(self) -> dict[str, object]:
        return {
            "quantity": (
                "tared contact wrench the environment applies to the held plug, gravity of hand "
                "and plug compensated: [Fx, Fy, Fz] N and [Tx, Ty, Tz] N m about the nominal plug tip"
            ),
            "frame": "nominal plug-tip frame (+z out of the socket; pressing into a face reads +Fz)",
            "rate_hz": self._adapter.control_frequency,
            "sensor_rate_hz": 1.0 / self._adapter.native_env.model.opt.timestep,
            "filter": (
                f"first-order low-pass at {self.lowpass_hz:g} Hz"
                if self.lowpass_hz > 0
                else "mean over each control period"
            ),
            "tare": "at reset, after settling (mean of 0.1 s)",
            "noise_std_n": self.noise_n,
            "bias_drift_n_per_min": self.bias_n_per_min,
            "torque_per_force_m": FT_TORQUE_PER_FORCE_M,
        }


class InsertionAdapter:
    """One PR insertion scene behind the ASPIRE environment boundary."""

    gripper_holds_intermediate = False
    BASE_FRAME = BASE_FRAME

    def __init__(
        self,
        *,
        registry: InsertionTaskRegistry | None = None,
        control_frequency: float = 20.0,
        camera_height: int = 360,
        camera_width: int = 480,
        horizon: int = 1200,
        run_observer: Any | None = None,
        perturbation_scale: float | None = None,
    ) -> None:
        if camera_height <= 0 or camera_width <= 0 or horizon <= 0:
            raise ValueError("camera dimensions and horizon must be positive")
        self.registry = registry or InsertionTaskRegistry()
        self.control_frequency = float(control_frequency)
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive and finite")
        self.control_period_s = 1.0 / self.control_frequency
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.horizon = int(horizon)
        if perturbation_scale is None:
            perturbation_scale = float(os.environ.get("INSERTION_PERTURBATION_SCALE", "1.0"))
        if not np.isfinite(perturbation_scale) or perturbation_scale < 0:
            raise ValueError("perturbation_scale must be finite and non-negative")
        self.perturbation_scale = float(perturbation_scale)
        #: Which error components a seed draws (INSERTION_PERTURBATION): "full", or
        #: "translation"/"rotation" only. A curriculum knob; the default is "full".
        self.perturbation_mode = os.environ.get("INSERTION_PERTURBATION", "full")
        if self.perturbation_mode not in {"full", "translation", "rotation"}:
            raise ValueError("INSERTION_PERTURBATION must be full, translation, or rotation")
        #: Re-render cameras only every N control ticks (INSERTION_CAMERA_REFRESH_TICKS);
        #: between renders an observation reuses the last images and their timestamp.
        self.camera_refresh_ticks = max(1, int(os.environ.get("INSERTION_CAMERA_REFRESH_TICKS", "1")))
        self._camera_cache: tuple[int, dict[str, CameraObservation]] | None = None
        #: Write the hidden perturbations and the true tip trace into the run directory
        #: (INSERTION_HOST_EVIDENCE=1). Off by default: run artifacts are what a debugging
        #: agent reads, and these are ground truth it must not have.
        self.host_evidence = os.environ.get("INSERTION_HOST_EVIDENCE", "0") == "1"
        self.start_lateral_m = float(os.environ.get("INSERTION_START_LATERAL_M", START_LATERAL_M))
        self.start_height_m = float(os.environ.get("INSERTION_START_HEIGHT_M", START_EXTRA_HEIGHT_M))
        self.start_tilt_rad = math.radians(
            float(os.environ.get("INSERTION_START_TILT_DEG", math.degrees(START_TILT_RAD)))
        )
        self.start_yaw_rad = math.radians(
            float(os.environ.get("INSERTION_START_YAW_DEG", math.degrees(START_YAW_RAD)))
        )
        #: Wrist force/torque (INSERTION_FORCE=1): insertion.get_wrench() and wrench(). Off by
        #: default, so programs see contact only as tracking error.
        self._ft = _WristForceSensor() if os.environ.get("INSERTION_FORCE", "0") == "1" else None
        self.controller = "JOINT_IMPEDANCE"
        self._run_observer = run_observer
        self._env: Any | None = None
        self._metadata: InsertionTaskMetadata | None = None
        self._task_context: TaskContext | None = None
        self._renderer: Any | None = None
        self._observation_cache: tuple[int, Observation] | None = None
        self._elapsed_steps = 0
        self._seed: int | None = None
        self._joint_target: np.ndarray | None = None
        self._ee_from_tip_nominal = np.eye(4)
        self._world_from_base = np.eye(4)
        self._seated_estimate_world = np.eye(4)
        self._episode_record: dict[str, object] = {}
        self.success_observed_step: int | None = None
        self._closed = False

    # ----------------------------------------------------------------- identity
    @property
    def embodiment(self) -> str:
        return EMBODIMENT

    @property
    def arms(self) -> tuple[str, ...]:
        return ("primary",)

    @property
    def current_time_s(self) -> float:
        return self._elapsed_steps * self.control_period_s

    @property
    def metadata(self) -> InsertionTaskMetadata:
        if self._metadata is None:
            raise RuntimeError("reset() must be called before task metadata is available")
        return self._metadata

    @property
    def native_env(self) -> Any:
        """Runtime-only insertion_skill InsertionEnv; never registered with programs."""
        return self._require_env()

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._require_env().manifest["arm_joints"])

    @property
    def episode_record(self) -> Mapping[str, object]:
        """Host-only record of this episode's hidden perturbations, for analysis."""
        return MappingProxyType(dict(self._episode_record))

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("reset() must be called before using the insertion adapter")
        return self._env

    # -------------------------------------------------------------------- reset
    def reset(
        self,
        task_ref: str | tuple[str, int] | InsertionTaskMetadata,
        seed: int,
    ) -> Observation:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        import mujoco
        from insertion_skill.env import InsertionEnv

        metadata = self.registry.resolve(task_ref)
        self._close_renderer()
        scene_dir = Path(metadata.scene_dir)
        env = InsertionEnv(scene_dir.name, envs=scene_dir.parent)
        model, data = env.model, env.data
        self._env = env
        self._metadata = metadata
        self._seed = seed
        self._closed = False
        self._elapsed_steps = 0
        self._observation_cache = None
        self.success_observed_step = None
        self._camera_cache = None
        self._tip_trace: list[list[float]] = []

        base_id = model.body(BASE_FRAME).id
        self._world_from_base = _transform(data.xmat[base_id], data.xpos[base_id])
        weld = model.eq("preheld_plug")
        self._weld_id = int(np.atleast_1d(weld.id)[0])
        self._ee_id = int(np.atleast_1d(weld.obj1id)[0])
        plug_id = model.body("plug").id
        ee_from_plug = _transform(_quat_to_matrix(weld.data[6:10]), weld.data[3:6])
        plug_from_tip = _transform(
            _quat_to_matrix(model.site_quat[env.tip]), model.site_pos[env.tip]
        )
        self._ee_from_tip_nominal = ee_from_plug @ plug_from_tip

        scale = self.perturbation_scale if seed > 0 else 0.0
        t_scale = 0.0 if self.perturbation_mode == "rotation" else scale
        r_scale = 0.0 if self.perturbation_mode == "translation" else scale
        rng = np.random.default_rng(seed)
        in_hand = _random_offset(
            rng,
            t_scale * IN_HAND_LATERAL_M,
            t_scale * IN_HAND_AXIAL_M,
            r_scale * IN_HAND_TILT_RAD,
            r_scale * IN_HAND_YAW_RAD,
        )
        socket_error = _random_offset(
            rng,
            t_scale * SOCKET_LATERAL_M,
            t_scale * SOCKET_AXIAL_M,
            r_scale * SOCKET_TILT_RAD,
            r_scale * SOCKET_YAW_RAD,
        )

        def draw_start() -> np.ndarray:
            offset = _random_offset(
                rng,
                scale * self.start_lateral_m,
                0.0,
                scale * self.start_tilt_rad,
                scale * self.start_yaw_rad,
            )
            offset[2, 3] = scale * rng.uniform(0.0, self.start_height_m)
            return offset

        start_offset = draw_start()

        # Move the weld so the plug really sits off the nominal grasp.
        ee_from_plug_actual = self._ee_from_tip_nominal @ in_hand @ invert_transform(plug_from_tip)
        model.eq_data[self._weld_id, 3:6] = ee_from_plug_actual[:3, 3]
        model.eq_data[self._weld_id, 6:10] = _matrix_to_quat(ee_from_plug_actual[:3, :3])
        joint = model.body_jntadr[plug_id]
        plug_qadr, plug_vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]

        def place(arm_joints: np.ndarray) -> None:
            """Set the arm, then the plug's free joint where the weld wants it."""
            data.qpos[env.qadr] = arm_joints
            data.qvel[:] = 0.0
            mujoco.mj_kinematics(model, data)
            world_from_ee = _transform(data.xmat[self._ee_id], data.xpos[self._ee_id])
            world_from_plug = world_from_ee @ ee_from_plug_actual
            data.qpos[plug_qadr : plug_qadr + 3] = world_from_plug[:3, 3]
            data.qpos[plug_qadr + 3 : plug_qadr + 7] = _matrix_to_quat(world_from_plug[:3, :3])
            mujoco.mj_forward(model, data)

        world_from_target = _transform(
            data.site_xmat[env.target], data.site_xpos[env.target]
        )
        self._seated_estimate_world = world_from_target @ socket_error
        start_joints = env.q
        backoff = 0.0
        if scale > 0.0:
            kinematics = InsertionKinematics(self)
            # The keyframe's nominal tip is at the true standoff pose; offset it from there.
            world_from_standoff = world_from_target.copy()
            world_from_standoff[:3, 3] += world_from_target[:3, 2] * (
                env.seated_depth + float(env.manifest.get("standoff_m", 0.01))
            )
            redraws = 0
            while True:
                lifted = world_from_standoff @ start_offset
                lifted[:3, 3] += world_from_target[:3, 2] * backoff
                solved, position_error, _ = kinematics.solve_world(lifted, env.q)
                if position_error > 1e-4:
                    # Unreachable start (only possible with a widened start range): redraw it
                    # from the same seeded generator, so the episode stays deterministic.
                    redraws += 1
                    if redraws > START_REDRAWS:
                        raise RuntimeError(f"no IK solution for the perturbed start (seed {seed})")
                    start_offset = draw_start()
                    backoff = 0.0
                    continue
                place(solved)
                if not self._plug_contacts():
                    break
                if backoff >= START_BACKOFF_MAX_M:
                    # Still touching after the full back-off (a widened, steeply tilted start):
                    # redraw rather than start the episode in contact.
                    redraws += 1
                    if redraws > START_REDRAWS:
                        break
                    start_offset = draw_start()
                    backoff = 0.0
                    continue
                backoff += START_BACKOFF_STEP_M
            start_joints = solved
        else:
            place(start_joints)

        self._joint_target = start_joints
        for _ in range(int(round(SETTLE_AFTER_RESET_S / model.opt.timestep))):
            env.step(self._impedance_torque(self._joint_target))
        if self._ft is not None:
            self._ft.reset(self, seed)

        from insertion_skill.validate_envs import penetrations

        self._episode_record = {
            "seed": seed,
            "perturbation_scale": scale,
            "perturbation_mode": self.perturbation_mode,
            "in_hand_offset_tip_frame": {
                "translation_m": in_hand[:3, 3].tolist(),
                "rotvec_rad": _matrix_to_rotvec(in_hand[:3, :3]).tolist(),
            },
            "socket_estimate_error_target_frame": {
                "translation_m": socket_error[:3, 3].tolist(),
                "rotvec_rad": _matrix_to_rotvec(socket_error[:3, :3]).tolist(),
            },
            "start_offset_nominal_tip_frame": {
                "translation_m": start_offset[:3, 3].tolist(),
                "rotvec_rad": _matrix_to_rotvec(start_offset[:3, :3]).tolist(),
            },
            "start_backoff_m": backoff,
            "start_redraws": redraws if scale > 0.0 else 0,
            "start_range": {
                "lateral_m": self.start_lateral_m,
                "height_m": self.start_height_m,
                "tilt_rad": self.start_tilt_rad,
                "yaw_rad": self.start_yaw_rad,
            },
            "start_tip_in_target_m": env.tip_in_target().tolist(),
            "start_penetrations": [list(map(str, item)) for item in penetrations(env)],
        }
        self._task_context = TaskContext(
            suite=metadata.suite_name,
            task_id=metadata.task_id,
            task_name=metadata.task_name,
            language=metadata.language,
            family=metadata.family,
            metadata={"task_ref": metadata.task_ref},
        )
        observation = self.get_observation()
        if self._run_observer is not None:
            record = metadata.to_manifest_record()
            if self.host_evidence:
                record["episode"] = dict(self._episode_record)
            self._run_observer.on_reset(observation, record)
        return observation

    def _plug_contacts(self) -> list[tuple[str, str]]:
        env = self._require_env()
        model, data = env.model, env.data
        plug_id = model.body("plug").id
        pairs = []
        for contact in data.contact[: data.ncon]:
            bodies = (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2])
            if plug_id in bodies or any(model.body(b).name.startswith("pr_") for b in bodies):
                pairs.append(tuple(model.body(b).name for b in bodies))
        return pairs

    # ------------------------------------------------------------------ control
    def _impedance_torque(self, joint_target: np.ndarray) -> np.ndarray:
        env = self._require_env()
        torque = env.gravity() + KP_JOINT * (joint_target - env.q) - KD_JOINT * env.dq
        ctrl_range = env.model.actuator_ctrlrange[:6]
        return np.clip(torque, ctrl_range[:, 0], ctrl_range[:, 1])

    def _joint_limits(self) -> np.ndarray:
        env = self._require_env()
        model = env.model
        return np.array([model.jnt_range[model.joint(name).id] for name in self.joint_names])

    def native_step(self, action: RobotAction) -> StepResult:
        env = self._require_env()
        unknown_arms = set(action.arms) - set(self.arms)
        if unknown_arms:
            return StepResult(
                ok=False,
                error=ApiError(ErrorCode.NOT_FOUND, f"unknown arms: {sorted(unknown_arms)}"),
            )
        command = action.arms.get("primary")
        if command is None:
            return StepResult(
                ok=False,
                error=ApiError(ErrorCode.INVALID_REQUEST, "action must command arm 'primary'"),
            )
        if command.embodiment != self.embodiment:
            return StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    f"action embodiment {command.embodiment!r} does not match {self.embodiment!r}",
                ),
            )
        if self._run_observer is not None:
            self._run_observer.before_step(action)
        try:
            limits = self._joint_limits()
            target = np.clip(np.asarray(command.target, dtype=np.float64), limits[:, 0], limits[:, 1])
            self._joint_target = target
            substeps = max(1, int(round(self.control_period_s / env.model.opt.timestep)))
            for _ in range(substeps):
                env.step(self._impedance_torque(target))
                if self._ft is not None:
                    self._ft.sample()
            if not bool(np.all(np.isfinite(env.data.qpos))):
                raise FloatingPointError("simulation diverged")
            if self._ft is not None:
                self._ft.end_period()
            self._elapsed_steps += 1
            self._observation_cache = None
            self._tip_trace.append([self._elapsed_steps, *env.tip_in_target().tolist()])
            if self.success_observed_step is None and env.seated():
                self.success_observed_step = self._elapsed_steps
            result = StepResult(
                ok=True,
                observation=self.get_observation(),
                diagnostics={"tip_in_target_m": env.tip_in_target().tolist()},
            )
        # Convert native failures into the public typed error result.
        except Exception as exc:
            result = StepResult(
                ok=False,
                error=ApiError(
                    ErrorCode.ADAPTER_FAILED,
                    f"insertion native step failed: {exc}",
                    details={"exception_type": type(exc).__name__},
                ),
            )
        if self._run_observer is not None:
            self._run_observer.after_step(action, result)
        return result

    def step(self, action: RobotAction) -> StepResult:
        runtime = self.native_step(action)
        return StepResult(
            ok=runtime.ok,
            observation=runtime.observation,
            terminated=runtime.terminated,
            truncated=runtime.truncated,
            error=runtime.public_view().error,
            diagnostics={
                "control_period_s": self.control_period_s,
                "step_index": self._elapsed_steps,
                **({"wrench_tip": self.wrench().tolist()} if runtime.ok and self._ft is not None else {}),
            },
        )

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        if isinstance(trajectory, SynchronizedTrajectory):
            raise ValueError("the insertion embodiment has one arm; use a Trajectory")
        if not isinstance(trajectory, Trajectory):
            raise ValueError("trajectory must be a Trajectory")
        if trajectory.embodiment != self.embodiment:
            raise ValueError(
                f"trajectory embodiment {trajectory.embodiment!r} does not match {self.embodiment!r}"
            )
        if trajectory.arm != "primary":
            raise ValueError("trajectory contains an unavailable arm")
        if not np.isclose(trajectory.dt_s, self.control_period_s, atol=1e-9, rtol=0.0):
            raise ValueError("trajectory dt_s must equal the insertion control period")
        state = self.get_robot_state()
        if tuple(trajectory.joint_names) != tuple(state.joint_names["primary"]):
            raise ValueError("trajectory joint_names do not match arm 'primary'")
        start_error = float(
            np.max(np.abs(trajectory.expected_start.joint_positions["primary"] - state.joint_positions["primary"]))
        )
        if start_error > TRAJECTORY_START_TOLERANCE_RAD:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.STALE_PLAN,
                    "trajectory start no longer matches measured robot state",
                    details={
                        "joint_error_rad": start_error,
                        "tolerance": TRAJECTORY_START_TOLERANCE_RAD,
                    },
                ),
                final_errors={"primary": start_error},
            )
        waypoints = np.asarray(trajectory.joint_positions, dtype=np.float64)
        final: Observation | None = None
        steps = 0

        def command(target: np.ndarray) -> RobotAction:
            return RobotAction(
                {"primary": ArmCommand("joint_position", target, embodiment=self.embodiment)}
            )

        def joint_error() -> float:
            assert final is not None
            return float(np.max(np.abs(final.robot_state.joint_positions["primary"] - waypoints[-1])))

        for waypoint in list(waypoints) + [waypoints[-1]] * TRAJECTORY_SETTLE_TICKS:
            if steps >= len(waypoints) and final is not None and joint_error() <= TRAJECTORY_TOLERANCE_RAD:
                break
            result = self.step(command(waypoint))
            if not result.ok or result.observation is None:
                return ExecutionResult(
                    ok=False, steps_executed=steps, final_observation=final, error=result.error
                )
            steps += 1
            final = result.observation
        error = joint_error()
        if error > TRAJECTORY_TOLERANCE_RAD:
            return ExecutionResult(
                ok=False,
                steps_executed=steps,
                final_observation=final,
                error=ApiError(
                    ErrorCode.TIMEOUT,
                    "final trajectory waypoint did not converge (blocked or still moving)",
                    details={"joint_error_rad": error, "tolerance_rad": TRAJECTORY_TOLERANCE_RAD},
                ),
                final_errors={"primary": error},
            )
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=final,
            final_errors={"primary": error},
            diagnostics={"joint_error_rad": error},
        )

    def _gripper_locked(self) -> ExecutionResult:
        return ExecutionResult(
            ok=False,
            steps_executed=0,
            error=ApiError(
                ErrorCode.UNSUPPORTED,
                "the plug is pre-grasped and the chopsticks are locked; the gripper is not commandable",
            ),
        )

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        del position, arm
        return self._gripper_locked()

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        del positions
        return self._gripper_locked()

    # ------------------------------------------------------------- observation
    def _tip_pose_in_base(self) -> Pose:
        env = self._require_env()
        data = env.data
        world_from_ee = _transform(data.xmat[self._ee_id], data.xpos[self._ee_id])
        base_from_tip = invert_transform(self._world_from_base) @ world_from_ee @ self._ee_from_tip_nominal
        return matrix_to_pose(base_from_tip, frame=BASE_FRAME)

    def get_robot_state(self) -> RobotState:
        env = self._require_env()
        return RobotState(
            joint_positions={"primary": env.q},
            joint_velocities={"primary": env.dq},
            end_effector_poses={"primary": self._tip_pose_in_base()},
            gripper_positions={"primary": 0.0},
            base_frame=BASE_FRAME,
            joint_names={"primary": self.joint_names},
            timestamp_s=self.current_time_s,
            embodiment=self.embodiment,
        )

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("reset() must be called before reading task context")
        return self._task_context

    def _camera_views(self) -> dict[str, tuple[Any, np.ndarray, float]]:
        """name -> (mujoco camera argument, world_from_camera in OpenGL axes, fovy deg)."""
        import mujoco

        env = self._require_env()
        model, data = env.model, env.data
        views: dict[str, tuple[Any, np.ndarray, float]] = {}
        overview = model.camera("overview").id
        views["overview"] = (
            "overview",
            _transform(data.cam_xmat[overview], data.cam_xpos[overview]),
            float(model.cam_fovy[overview]),
        )
        # Close-up: looks at the socket mouth from 40 degrees off the insertion axis. It is aimed
        # at the socket *estimate* (what a real cell would point a camera at), never at the true
        # socket site, so the socket's place in the image carries no ground truth.
        estimate = self._seated_estimate_world
        out = estimate[:3, 2]
        lookat = estimate[:3, 3] + out * (env.seated_depth + 0.004)
        side = np.cross(out, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side) < 0.3:
            side = np.cross(out, np.array([1.0, 0.0, 0.0]))
        side /= np.linalg.norm(side)
        up = np.cross(side, out)
        direction = math.cos(math.radians(40)) * out + math.sin(math.radians(40)) * (0.6 * side + 0.8 * up)
        position = lookat + 0.16 * direction / np.linalg.norm(direction)
        forward = lookat - position
        distance = float(np.linalg.norm(forward))
        forward /= distance
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
        camera.elevation = math.degrees(math.asin(np.clip(forward[2], -1.0, 1.0)))
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)
        rotation = np.column_stack([right, camera_up, -forward])
        views["closeup"] = (camera, _transform(rotation, position), float(model.vis.global_.fovy))
        return views

    def _render_cameras(self) -> dict[str, CameraObservation]:
        import mujoco

        env = self._require_env()
        if self._renderer is None:
            width = min(self.camera_width, int(env.model.vis.global_.offwidth))
            height = min(self.camera_height, int(env.model.vis.global_.offheight))
            self._renderer = mujoco.Renderer(env.model, height, width)
        renderer = self._renderer
        base_from_world = invert_transform(self._world_from_base)
        image_from_opengl = np.diag((1.0, -1.0, -1.0, 1.0))
        # Sites mark ground truth (socket_frame, insert_target, plug_tip); never draw them.
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        cameras: dict[str, CameraObservation] = {}
        for name, (camera, world_from_camera, fovy) in self._camera_views().items():
            renderer.disable_depth_rendering()
            renderer.update_scene(env.data, camera=camera, scene_option=options)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(env.data, camera=camera, scene_option=options)
            depth = np.asarray(renderer.render(), dtype=np.float64).copy()
            renderer.disable_depth_rendering()
            focal = 0.5 * rgb.shape[0] / math.tan(math.radians(fovy) / 2.0)
            intrinsics = np.array(
                [[focal, 0.0, 0.5 * rgb.shape[1]], [0.0, focal, 0.5 * rgb.shape[0]], [0.0, 0.0, 1.0]]
            )
            cameras[name] = CameraObservation(
                rgb=rgb,
                depth_m=np.clip(depth, 0.0, None),
                intrinsics=intrinsics,
                frame=f"camera/{name}",
                camera_pose=matrix_to_pose(
                    base_from_world @ world_from_camera @ image_from_opengl, frame=BASE_FRAME
                ),
                timestamp_s=self.current_time_s,
            )
        return cameras

    def get_observation(self) -> Observation:
        cached = self._observation_cache
        if cached is not None and cached[0] == self._elapsed_steps:
            return cached[1]
        cameras = self._camera_cache
        if cameras is None or self._elapsed_steps - cameras[0] >= self.camera_refresh_ticks:
            cameras = (self._elapsed_steps, self._render_cameras())
            self._camera_cache = cameras
        observation = Observation(
            cameras=cameras[1],
            robot_state=self.get_robot_state(),
            task_context=self.get_task_context(),
            timestamp_s=self.current_time_s,
        )
        self._observation_cache = (self._elapsed_steps, observation)
        return observation

    # ----------------------------------------------------------------- metadata
    def _plug_extent_in_tip(self) -> list[float]:
        """Half-extents of the plug's collision geoms, measured along the nominal tip axes."""
        env = self._require_env()
        model = env.model
        plug_id = model.body("plug").id
        plug_from_tip = _transform(_quat_to_matrix(model.site_quat[env.tip]), model.site_pos[env.tip])
        tip_from_plug = invert_transform(plug_from_tip)
        points = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != plug_id or model.geom_contype[geom] == 0:
                continue
            center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
            plug_from_geom = _transform(_quat_to_matrix(model.geom_quat[geom]), model.geom_pos[geom])
            for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                local = np.r_[center + corner * half, 1.0]
                points.append((tip_from_plug @ plug_from_geom @ local)[:3])
        if not points:
            return [0.0, 0.0, 0.0]
        points_array = np.asarray(points)
        return [float(value) for value in points_array.min(axis=0)] + [
            float(value) for value in points_array.max(axis=0)
        ]

    def get_task_metadata(self) -> Mapping[str, object]:
        env = self._require_env()
        manifest = env.manifest
        base_from_world = invert_transform(self._world_from_base)
        vision = self._vision_socket_estimate()
        seated = base_from_world @ (vision["seated_world"] if vision else self._seated_estimate_world)
        mouth = seated.copy()
        mouth[:3, 3] = seated[:3, 3] + seated[:3, 2] * env.seated_depth
        extent = self._plug_extent_in_tip()
        record = self.metadata.to_manifest_record()
        record.update(
            {
                "base_frame": BASE_FRAME,
                "tip_frame": (
                    "end_effector_poses['primary'] is the plug tip as the nominal grasp places it: "
                    "+z points back out of the socket (opposite the insertion direction), so a "
                    "perfectly seated plug has its tip frame equal to seated_tip_pose_estimate"
                ),
                "seated_tip_pose_estimate": _pose_record(matrix_to_pose(seated, frame=BASE_FRAME)),
                "socket_mouth_pose_estimate": _pose_record(matrix_to_pose(mouth, frame=BASE_FRAME)),
                "insertion_direction": [float(value) for value in -seated[:3, 2]],
                "seated_depth_m": float(env.seated_depth),
                "standoff_m": float(manifest.get("standoff_m", 0.01)),
                "success_tolerance_m": {
                    "axial": float(env.success_tol[0]),
                    "lateral": float(env.success_tol[1]),
                },
                "plug_bounds_in_tip_frame_m": {"min": extent[:3], "max": extent[3:]},
                "grasp": {
                    "type": manifest["grasp"]["type"],
                    "jaw_gap_m": float(manifest["grasp"]["jaw_gap_m"]),
                },
                "uncertainty": self._uncertainty(),
                "sensing": (
                    "wrist force/torque: insertion.get_wrench() returns the tared contact wrench "
                    "on the plug in the nominal tip frame"
                    if self._ft is not None
                    else "no force/torque sensing; contact is visible only as joint tracking error"
                ),
            }
        )
        if vision:
            # The socket pose and its uncertainty are what the cameras measured (perception.py).
            bounds = vision["bounds"]
            record["uncertainty"] = dict(
                record["uncertainty"],
                socket_estimate_lateral_m=float(bounds["lateral_m"]),
                socket_estimate_axial_m=float(bounds["axial_m"]),
                socket_estimate_tilt_rad=float(bounds["tilt_rad"]),
                socket_estimate_yaw_rad=float(bounds["yaw_rad"]),
            )
            record["socket_estimate_source"] = {
                "source": "vision",
                "status": vision["status"],
                "timing_s": round(float(vision["timing"]["total_s"]), 3),
            }
        return MappingProxyType(record)

    def _vision_socket_estimate(self) -> Mapping[str, Any] | None:
        """INSERTION_SOCKET_SOURCE=vision: the camera-based socket estimate, once per episode."""
        if os.environ.get("INSERTION_SOCKET_SOURCE", "synthetic") != "vision":
            return None
        env = self._require_env()
        cached = getattr(self, "_vision_cache", None)
        if cached is None or cached[0] is not env:
            from cap_harness.insertion import perception

            estimate = perception.estimate_socket(self)
            self._vision_cache = (env, estimate)
            self._episode_record["socket_vision"] = {
                key: estimate[key] for key in ("status", "bounds", "timing", "attempts", "diagnostics") if key in estimate
            }
        return self._vision_cache[1]

    def _uncertainty(self) -> dict[str, float]:
        """Bounds of the hidden errors, as a perception/grasp spec sheet would state them."""
        t_scale = 0.0 if self.perturbation_mode == "rotation" else self.perturbation_scale
        r_scale = 0.0 if self.perturbation_mode == "translation" else self.perturbation_scale
        return {
            "socket_estimate_lateral_m": SOCKET_LATERAL_M * t_scale,
            "socket_estimate_axial_m": SOCKET_AXIAL_M * t_scale,
            "socket_estimate_tilt_rad": SOCKET_TILT_RAD * r_scale,
            "socket_estimate_yaw_rad": SOCKET_YAW_RAD * r_scale,
            "grasp_lateral_m": IN_HAND_LATERAL_M * t_scale,
            "grasp_axial_m": IN_HAND_AXIAL_M * t_scale,
            "grasp_tilt_rad": IN_HAND_TILT_RAD * r_scale,
            "grasp_yaw_rad": IN_HAND_YAW_RAD * r_scale,
        }

    def get_controller_metadata(self) -> Mapping[str, object]:
        limits = self._joint_limits()
        speed = tuple(float(MAX_JOINT_SPEED_RAD_S) for _ in self.joint_names)
        return MappingProxyType(
            {
                "action_dimension": len(self.joint_names),
                "action_lower_bounds": tuple(-value for value in speed),
                "action_upper_bounds": speed,
                "action_bounds_semantics": "max joint speed (rad/s) used to space interpolated waypoints",
                "arm_names": self.arms,
                "base_frame": BASE_FRAME,
                "camera_names": self.metadata.camera_names,
                "control_frequency_hz": self.control_frequency,
                "control_period_s": self.control_period_s,
                "controller": self.controller,
                "joint_names": self.joint_names,
                "joint_limits_rad": tuple(tuple(float(v) for v in row) for row in limits),
                "joint_stiffness_nm_per_rad": tuple(float(v) for v in KP_JOINT),
                "joint_damping_nms_per_rad": tuple(float(v) for v in KD_JOINT),
                "gripper_semantics": {"normalized": "locked: the plug is pre-grasped"},
                "controllable_gripper_arms": (),
                "supported_action_modes": ("joint_position",),
                "trajectory_tolerance_rad": TRAJECTORY_TOLERANCE_RAD,
                "trajectory_settle_ticks": TRAJECTORY_SETTLE_TICKS,
                **({"force_torque": self._ft.metadata()} if self._ft is not None else {}),
            }
        )

    # ------------------------------------------------------------------- force
    @property
    def force_enabled(self) -> bool:
        return self._ft is not None

    def wrench(self) -> np.ndarray:
        """Contact wrench on the plug, [Fx, Fy, Fz, Tx, Ty, Tz] (N, N m), in the nominal tip frame.

        Tared at reset and latched once per control period; needs INSERTION_FORCE=1.
        """
        if self._ft is None:
            raise RuntimeError("wrist force/torque is disabled; set INSERTION_FORCE=1")
        self._require_env()
        return self._ft.wrench

    def _get_wrench(self) -> Mapping[str, object]:
        """Tared contact wrench the socket applies to the held plug, at the nominal plug tip.

        Returns ``force_n`` and ``torque_nm`` (3 values each) in the nominal plug-tip frame,
        the frame of ``end_effector_poses['primary']``: +z points out of the socket, so pushing
        the plug into a face reads positive ``force_n[2]``. The hand and plug weight are
        compensated and the reading is tared at reset, so free motion reads about zero. It is
        updated once per control period (the mean over the period, or a low-pass filter).
        """
        wrench = self.wrench()
        return MappingProxyType(
            {
                "force_n": [float(value) for value in wrench[:3]],
                "torque_nm": [float(value) for value in wrench[3:]],
                "frame": "tip",
                "timestamp_s": self.current_time_s,
            }
        )

    @property
    def get_wrench(self) -> Any:
        """The ``insertion.get_wrench`` tool, or None (not registered) when force is disabled."""
        return self._get_wrench if self._ft is not None else None

    def get_planning_context(self) -> RobotPlanningContext:
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model="pr_v4p1sr_chopsticks",
            joint_names={"primary": self.joint_names},
            base_transforms={"primary": np.eye(4, dtype=np.float64)},
            end_effector_links={"primary": EE_BODY},
        )

    # ------------------------------------------------------------------ scoring
    def check_success(self) -> bool:
        return bool(self._require_env().seated())

    def tip_error_in_target(self) -> list[float]:
        """Host-only: true plug-tip position in the seated-target frame (m)."""
        return [float(value) for value in self._require_env().tip_in_target()]

    def finish_episode_recording(self) -> None:
        """Write host-only evaluation evidence (true tip error, hidden perturbations).

        The run recorder calls this before it seals the run manifest. Nothing here is
        visible to the generated program; it is for whoever debugs the run.
        """
        root = getattr(self._run_observer, "root", None)
        if root is None or self._env is None or not self.host_evidence:
            return
        import json

        env = self._require_env()
        tip = env.tip_in_target()
        record = {
            "task_success": bool(env.seated()),
            "success_observed_step": self.success_observed_step,
            "steps": self._elapsed_steps,
            "final_tip_in_target_m": tip.tolist(),
            "final_axial_error_m": float(tip[2]),
            "final_lateral_error_m": float(np.hypot(tip[0], tip[1])),
            "success_tolerance_m": env.success_tol.tolist(),
            "episode": dict(self._episode_record),
            "tip_trace_in_target_m": [
                [int(row[0])] + [round(value, 6) for value in row[1:]] for row in self._tip_trace
            ],
        }
        path = Path(root) / "evaluation" / "insertion.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- lifecycle
    def _close_renderer(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
        self._renderer = None

    def close(self) -> None:
        if self._closed:
            return
        self._close_renderer()
        self._env = None
        self._closed = True

    def kinematics(self) -> InsertionKinematics:
        """An IK provider bound to this episode's model and nominal grasp."""
        return InsertionKinematics(self)


class InsertionKinematics:
    """Damped least-squares IK for the nominal plug-tip frame, on a private MjData copy."""

    MAX_ITERATIONS = 300
    POSITION_TOLERANCE_M = 1e-6
    ROTATION_TOLERANCE_RAD = 1e-5

    def __init__(self, adapter: InsertionAdapter) -> None:
        import mujoco

        env = adapter.native_env
        self._adapter = adapter
        self._model = env.model
        self._data = mujoco.MjData(env.model)
        self._qadr = env.qadr
        self._vadr = env.vadr
        self._ee_id = adapter._ee_id
        self._ee_from_tip = adapter._ee_from_tip_nominal.copy()
        self._world_from_base = adapter._world_from_base.copy()
        self._limits = adapter._joint_limits()

    def _tip_world(self, joints: np.ndarray) -> np.ndarray:
        import mujoco

        self._data.qpos[self._qadr] = joints
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)  # mj_jac needs cdof and subtree_com
        world_from_ee = _transform(self._data.xmat[self._ee_id], self._data.xpos[self._ee_id])
        return world_from_ee @ self._ee_from_tip

    def solve_ik(self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary") -> IKResult:
        if arm != "primary":
            return IKResult(
                ok=False,
                arm=arm,
                embodiment=EMBODIMENT,
                error=ApiError(ErrorCode.NOT_FOUND, f"unknown arm {arm!r}"),
            )
        if target_pose.frame != BASE_FRAME:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment=EMBODIMENT,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    f"target pose must be in {BASE_FRAME!r}, got {target_pose.frame!r}",
                ),
            )
        target = self._world_from_base @ np.asarray(target_pose.as_matrix())
        joints, position_error, rotation_error = self.solve_world(
            target, np.asarray(robot_state.joint_positions[arm], dtype=np.float64)
        )
        if position_error > 1e-4 or rotation_error > 1e-3:
            return IKResult(
                ok=False,
                arm=arm,
                embodiment=EMBODIMENT,
                error=ApiError(
                    ErrorCode.IK_FAILED,
                    "no joint solution reaches the requested plug-tip pose within joint limits",
                    details={
                        "position_error_m": position_error,
                        "rotation_error_rad": rotation_error,
                    },
                ),
            )
        return IKResult(
            ok=True,
            joint_positions=joints,
            arm=arm,
            embodiment=EMBODIMENT,
            diagnostics={
                "position_error_m": position_error,
                "rotation_error_rad": rotation_error,
                "solver": "mujoco-dls",
            },
        )

    def solve_world(
        self, target: np.ndarray, seed_joints: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Joints placing the nominal tip at ``target`` (world 4x4), with residual errors."""
        import mujoco

        joints = np.array(seed_joints, dtype=np.float64)
        self._data.qpos[:] = self._adapter.native_env.data.qpos
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        damping = 1e-4
        position_error = rotation_error = float("inf")
        for _ in range(self.MAX_ITERATIONS):
            tip = self._tip_world(joints)
            dp = target[:3, 3] - tip[:3, 3]
            dr = _matrix_to_rotvec(target[:3, :3] @ tip[:3, :3].T)
            position_error, rotation_error = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            if position_error < self.POSITION_TOLERANCE_M and rotation_error < self.ROTATION_TOLERANCE_RAD:
                break
            mujoco.mj_jac(self._model, self._data, jacp, jacr, tip[:3, 3], self._ee_id)
            jacobian = np.vstack([jacp[:, self._vadr], jacr[:, self._vadr]])
            error = np.r_[dp, dr]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
            norm = float(np.linalg.norm(delta))
            if norm > 0.2:
                delta *= 0.2 / norm
            joints = np.clip(joints + delta, self._limits[:, 0], self._limits[:, 1])
        return joints, position_error, rotation_error


__all__ = ["BASE_FRAME", "EMBODIMENT", "InsertionAdapter", "InsertionKinematics"]
