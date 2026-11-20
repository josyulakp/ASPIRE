"""Models of the real PR arm's controllers, for training policies that transfer to the robot.

The adapter's default ``JOINT_IMPEDANCE`` (gravity compensation + soft joint PD) does not exist on the real arm:
pr_hardware exposes joint velocity (/right/move), Cartesian EE velocity through diff-IK (/right/command_ee_vel),
stiff joint position (/right/pos_move) and gravity-compensation torque only. INSERTION_CONTROLLER selects:

``admittance``  the stack the real skill will use: an F/T admittance law over the commanded tip pose, a pose-tracking
                EE twist, damped-least-squares diff-IK to joint velocities, and a joint velocity servo model
                (command delay, first-order lag, velocity/acceleration limits, stiff tracking of the integrated
                reference: an industrial velocity loop).
``position``    /right/pos_move: the program's joint target, rate-limited, tracked by the same stiff servo.

Every control period the program (or policy) gives one joint-position target, as with the impedance controller. Its
forward kinematics through the nominal grasp is the tip *setpoint*. The admittance frame is the nominal plug-tip
frame, so the measured wrench (the zeroed contact wrench on the plug, as from the Kunwei F/T after set_zero and
gravity compensation) is expressed where the compliance acts: +z out of the socket.

    M e'' + D e' + K e = F_ext        e: compliant offset of the reference from the setpoint (tip frame, 6-D)
    twist = d/dt(reference) + K_track * (reference - measured tip)        (world frame, clipped)
    qd_cmd = J^T (J J^T + lambda^2 I)^-1 twist                           (tip Jacobian, 6 arm joints)
    qd_ref = lag(delay(clip(qd_cmd)))   q_ref += qd_ref dt   tau = g + Kp (q_ref - q) + Kd (qd_ref - qd)

All numbers are placeholders until the real arm is identified (step responses on /right/move and
/right/command_ee_vel, F/T-to-command latency): set them with INSERTION_CONTROLLER_PARAMS=<json file>, whose keys
override DEFAULTS. ``randomize`` scales the identified servo and admittance values per episode (domain
randomisation; 0 = off).
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Any

import numpy as np

DEFAULTS: dict[str, Any] = {
    # Inner loop (diff-IK + admittance) rate; the simulator runs at 2 kHz, so this is every 4th substep.
    "inner_rate_hz": 500.0,
    # Joint velocity servo: command delay, first-order lag, limits, and stiff tracking of the integrated reference.
    "delay_s": 0.006,
    "lag_s": 0.02,
    "qd_max": [1.0, 1.0, 2.0, 2.0, 2.0, 2.0],            # URDF velocity limits (rad/s), arm joints 1 2 3 4 6 7
    "qdd_max": 10.0,                                     # rad/s^2
    # Servo stiffness as a bandwidth: kp = I w^2, kd = 2 zeta I w per joint, from the joint-space inertia at reset
    # (gains above ~0.3 / timestep for the damping chatter at the 0.5 ms step). servo_kp / servo_kd override.
    "servo_bandwidth_hz": 30.0,
    "servo_damping_ratio": 1.0,
    "servo_kp": None,
    "servo_kd": None,
    # Admittance in the tip frame [x y z rx ry rz]; damping from the ratio unless given.
    "adm_mass": [5.0, 5.0, 5.0, 0.05, 0.05, 0.05],         # above the hand inertia the F/T sees
    "adm_stiffness": [500.0, 500.0, 1000.0, 2.0, 2.0, 0.5], # soft laterally so chamfers guide the plug
    "adm_damping_ratio": 10.0,                          # overdamped: lower values bounce on stiff contact (6 Hz
                                                         # impacts at the grip friction limit, plug slips in the jaws)
    "adm_damping": None,
    "adm_max_offset": [0.03, 0.03, 0.03, 0.2, 0.2, 0.5],  # m, rad: the compliant offset saturates here
    # Force limit (tip frame, N): above it the arm backs off along the force at force_limit_gain (m/s per N of
    # excess), on top of the admittance, at the inner-loop rate (the admittance's M is too slow for impacts).
    # The setpoint is also kept within force_limit / K of the measured tip, so the steady force stays under the
    # limit and the offset never saturates into stiff pushing. None disables. Above force_abort (N, |F|) the
    # controller latches a stop: it holds the measured tip backed off 2 mm along +z (the real stack's watchdog).
    "force_limit": [15.0, 15.0, 25.0],
    "force_limit_gain": 0.004,
    "force_abort": 60.0,
    "ft_lowpass_hz": 30.0,                               # the controller's own F/T filter
    "ft_deadband": [0.3, 0.3, 0.3, 0.005, 0.005, 0.005], # N, N m: below this the admittance ignores the wrench
    # Pose tracking and diff-IK.
    "track_gain": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],  # 1/s
    "v_max": 0.05,                                       # m/s
    "w_max": 0.5,                                        # rad/s
    "dls_lambda": 0.02,
    # /right/pos_move model: the target is approached at this joint speed, then held by the stiff servo.
    "position_qd_max": 0.5,
    # Per-episode domain randomisation: multiplicative +-randomize on delay, lag, servo gains, admittance K and M.
    "randomize": 0.0,
}


def load_params() -> dict[str, Any]:
    params = dict(DEFAULTS)
    path = os.environ.get("INSERTION_CONTROLLER_PARAMS")
    if path:
        overrides = json.loads(open(path).read())
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown controller parameters: {sorted(unknown)}")
        params.update(overrides)
    return params


def _rotvec(rotation: np.ndarray) -> np.ndarray:
    cos = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    if angle < 1e-7:
        return 0.5 * axis
    return axis * (angle / (2.0 * math.sin(angle)))


def _exp(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    k = rotvec / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


class ArmController:
    """Per-substep torques for the ``admittance`` or ``position`` controller of one episode."""

    def __init__(self, adapter: Any, mode: str, params: dict[str, Any] | None = None) -> None:
        if mode not in {"admittance", "position"}:
            raise ValueError(f"INSERTION_CONTROLLER={mode!r}; expected impedance, admittance or position")
        self.mode = mode
        self.base_params = dict(params or load_params())
        self.adapter = adapter

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int) -> None:
        """Bind to the episode after the reset has settled: reference = where the arm is, nothing in flight."""
        import mujoco

        adapter = self.adapter
        env = adapter.native_env
        self.env, self.model, self.data = env, env.model, env.data
        self.dt = float(self.model.opt.timestep)
        p = dict(self.base_params)
        rng = np.random.default_rng([seed, 0xC7])
        r = float(p["randomize"])

        def jitter(value):
            value = np.asarray(value, dtype=np.float64)
            return value * (1.0 + rng.uniform(-r, r, size=value.shape)) if r > 0 else value

        self.every = max(1, int(round(1.0 / (p["inner_rate_hz"] * self.dt))))
        self.dt_in = self.every * self.dt
        self.delay_ticks = max(0, int(round(float(jitter(p["delay_s"])) / self.dt_in)))
        self.lag = float(jitter(p["lag_s"]))
        self.qd_max = np.asarray(p["qd_max"], dtype=np.float64)
        self.qdd_max = float(p["qdd_max"])
        full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, full)
        inertia = np.diag(full)[env.vadr]
        omega = 2.0 * math.pi * float(jitter(p["servo_bandwidth_hz"]))
        zeta = float(p["servo_damping_ratio"])
        self.kp = jitter(p["servo_kp"]) if p["servo_kp"] is not None else inertia * omega**2
        self.kd = jitter(p["servo_kd"]) if p["servo_kd"] is not None else 2.0 * zeta * inertia * omega
        self.mass = jitter(p["adm_mass"])
        self.stiff = jitter(p["adm_stiffness"])
        if p["adm_damping"] is not None:
            self.damp = np.asarray(p["adm_damping"], dtype=np.float64)
        else:
            self.damp = 2.0 * float(p["adm_damping_ratio"]) * np.sqrt(self.mass * self.stiff)
        self.max_offset = np.asarray(p["adm_max_offset"], dtype=np.float64)
        self.ft_alpha = 1.0 - math.exp(-2.0 * math.pi * float(p["ft_lowpass_hz"]) * self.dt)
        self.deadband = np.asarray(p["ft_deadband"], dtype=np.float64)
        self.track = np.asarray(p["track_gain"], dtype=np.float64)
        self.v_max, self.w_max = float(p["v_max"]), float(p["w_max"])
        self.lam2 = float(p["dls_lambda"]) ** 2
        self.position_qd_max = float(p["position_qd_max"])
        self.force_limit = None if p["force_limit"] is None else np.asarray(p["force_limit"], dtype=np.float64)
        self.force_gain = float(p["force_limit_gain"])
        self.force_abort = None if p["force_abort"] is None else float(p["force_abort"])
        self.aborted_at: float | None = None
        self.peak_force = 0.0
        self.capped_cycles = 0
        self.params = p
        self.limits = adapter._joint_limits()
        self.kin = adapter.kinematics()
        self.ctrl_range = self.model.actuator_ctrlrange[:6]

        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))
        self._mujoco = mujoco
        self.q_ref = env.q.copy()
        self.qd_ref = np.zeros(6)
        self.qd_lagged = np.zeros(6)
        fill = env.q.copy() if self.mode == "position" else np.zeros(6)
        self.queue: deque[np.ndarray] = deque([fill.copy() for _ in range(self.delay_ticks)])
        self.offset = np.zeros(6)          # compliant offset e (tip-setpoint frame)
        self.offset_rate = np.zeros(6)
        self.wrench = np.zeros(6)          # filtered contact wrench, nominal tip frame
        self.setpoint = self.tip_world()
        self.joint_target = env.q.copy()
        self.substep = 0
        # Wrench channel for the admittance: the adapter's F/T model (same noise/bias knobs), tared now.
        from cap_harness.insertion.adapter import _WristForceSensor

        self.sensor = _WristForceSensor()
        self.sensor.reset(adapter, seed)

    # ------------------------------------------------------------------ kinematics
    def tip_world(self) -> np.ndarray:
        data, ee = self.data, self.adapter._ee_id
        world_from_ee = np.eye(4)
        world_from_ee[:3, :3] = data.xmat[ee].reshape(3, 3)
        world_from_ee[:3, 3] = data.xpos[ee]
        return world_from_ee @ self.adapter._ee_from_tip_nominal

    def tip_jacobian_world(self, tip: np.ndarray) -> np.ndarray:
        self._mujoco.mj_jac(self.model, self.data, self._jp, self._jr, tip[:3, 3], self.adapter._ee_id)
        vadr = self.env.vadr
        return np.vstack([self._jp[:, vadr], self._jr[:, vadr]])

    # ------------------------------------------------------------------ control
    def set_target(self, joint_target: np.ndarray, setpoint_world: np.ndarray) -> None:
        """New control period: the commanded joint target and its tip pose (FK through the nominal grasp)."""
        self.joint_target = np.asarray(joint_target, dtype=np.float64).copy()
        self.setpoint = np.asarray(setpoint_world, dtype=np.float64).copy()

    def _contact_wrench_tip(self) -> np.ndarray:
        sensor = self.sensor
        return sensor._to_tip(sensor._contact_sample())

    def _reference(self, setpoint: np.ndarray) -> np.ndarray:
        """Setpoint displaced by the compliant offset, offset expressed in the setpoint's tip frame."""
        reference = setpoint.copy()
        rotation = setpoint[:3, :3]
        reference[:3, 3] = setpoint[:3, 3] + rotation @ self.offset[:3]
        reference[:3, :3] = rotation @ _exp(self.offset[3:])
        return reference

    def _safe_setpoint(self, tip: np.ndarray) -> np.ndarray:
        """The setpoint after the force cap and the abort latch."""
        setpoint = self.setpoint.copy()
        rotation = setpoint[:3, :3]
        force = float(np.linalg.norm(self.wrench[:3]))
        self.peak_force = max(self.peak_force, force)
        if self.aborted_at is None and self.force_abort is not None and force > self.force_abort:
            self.aborted_at = float(self.data.time)
            self.abort_pose = tip.copy()
            self.abort_pose[:3, 3] += tip[:3, :3] @ np.array([0.0, 0.0, 0.002])
        if self.aborted_at is not None:
            return self.abort_pose.copy()
        if self.force_limit is not None:
            lead = rotation.T @ (setpoint[:3, 3] - tip[:3, 3])
            cap = self.force_limit / self.stiff[:3]
            clipped = np.clip(lead, -cap, cap)
            if np.any(clipped != lead):
                self.capped_cycles += 1
                setpoint[:3, 3] = tip[:3, 3] + rotation @ clipped
        return setpoint

    def _inner(self) -> None:
        """One diff-IK / admittance cycle: update the joint velocity command."""
        dt = self.dt_in
        tip = self.tip_world()
        if self.mode == "admittance":
            # The wrench is in the current tip frame; express it in the setpoint frame (they differ by the offset).
            w_tip = self.wrench.copy()
            w_tip = np.where(np.abs(w_tip) > self.deadband, w_tip - np.sign(w_tip) * self.deadband, 0.0)
            setpoint = self._safe_setpoint(tip)
            setpoint_from_tip = setpoint[:3, :3].T @ tip[:3, :3]
            force = np.r_[setpoint_from_tip @ w_tip[:3], setpoint_from_tip @ w_tip[3:]]
            accel = (force - self.damp * self.offset_rate - self.stiff * self.offset) / self.mass
            self.offset_rate = self.offset_rate + accel * dt
            self.offset = self.offset + self.offset_rate * dt
            clipped = np.clip(self.offset, -self.max_offset, self.max_offset)
            self.offset_rate = np.where(clipped != self.offset, 0.0, self.offset_rate)
            self.offset = clipped
            reference = self._reference(setpoint)
            # Feed-forward of the admittance's own motion (setpoint steps arrive at 20 Hz; tracking handles them).
            rotation = setpoint[:3, :3]
            ff = np.r_[rotation @ self.offset_rate[:3], rotation @ self.offset_rate[3:]]
            error = np.r_[reference[:3, 3] - tip[:3, 3], _rotvec(reference[:3, :3] @ tip[:3, :3].T)]
            twist = ff + self.track * error
            speed, spin = np.linalg.norm(twist[:3]), np.linalg.norm(twist[3:])
            if speed > self.v_max:
                twist[:3] *= self.v_max / speed
            if spin > self.w_max:
                twist[3:] *= self.w_max / spin
            if self.force_limit is not None and self.aborted_at is None:
                excess = np.sign(force[:3]) * np.maximum(np.abs(force[:3]) - self.force_limit, 0.0)
                if np.any(excess != 0.0):
                    self.capped_cycles += 1
                    back = np.clip(self.force_gain * excess, -2 * self.v_max, 2 * self.v_max)
                    twist[:3] += rotation @ back
            jac = self.tip_jacobian_world(tip)
            qd_cmd = jac.T @ np.linalg.solve(jac @ jac.T + self.lam2 * np.eye(6), twist)
        else:
            # pos_move: the (delayed) joint target is approached at a capped joint speed, then held; the servo's own
            # position loop closes this, so no velocity lag applies.
            self.queue.append(self.joint_target.copy())
            target = self.queue.popleft() if len(self.queue) > self.delay_ticks else self.q_ref
            step = np.clip(target - self.q_ref, -self.position_qd_max * dt, self.position_qd_max * dt)
            self.qd_lagged = step / dt
            return
        # Joint velocity servo: limits, command delay, first-order lag.
        scale = np.max(np.abs(qd_cmd) / self.qd_max)
        if scale > 1.0:
            qd_cmd = qd_cmd / scale
        self.queue.append(qd_cmd)
        delayed = self.queue.popleft()
        alpha = 1.0 if self.lag <= 0 else 1.0 - math.exp(-dt / self.lag)
        lagged = self.qd_lagged + alpha * (delayed - self.qd_lagged)
        lagged = self.qd_lagged + np.clip(lagged - self.qd_lagged, -self.qdd_max * dt, self.qdd_max * dt)
        self.qd_lagged = lagged

    def torque(self) -> np.ndarray:
        """Arm torques for the next simulator substep."""
        self.wrench += self.ft_alpha * (self._contact_wrench_tip() - self.wrench)
        if self.substep % self.every == 0:
            self._inner()
        self.substep += 1
        qd = self.qd_lagged
        self.q_ref = self.q_ref + qd * self.dt
        at_limit = (self.q_ref <= self.limits[:, 0]) | (self.q_ref >= self.limits[:, 1])
        self.q_ref = np.clip(self.q_ref, self.limits[:, 0], self.limits[:, 1])
        qd = np.where(at_limit, 0.0, qd)
        torque = self.env.gravity() + self.kp * (self.q_ref - self.env.q) + self.kd * (qd - self.env.dq)
        return np.clip(torque, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def metadata(self) -> dict[str, Any]:
        return {"mode": self.mode, "params": self.params,
                "episode": {"delay_ticks": self.delay_ticks, "inner_dt_s": self.dt_in, "lag_s": self.lag,
                            "adm_stiffness": self.stiff.tolist(), "adm_mass": self.mass.tolist(),
                            "adm_damping": self.damp.tolist(), "servo_kp": self.kp.tolist(),
                            "servo_kd": self.kd.tolist(), "peak_force_n": round(self.peak_force, 2),
                            "force_capped_cycles": self.capped_cycles, "aborted_at_s": self.aborted_at}}
