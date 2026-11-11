"""Insertion embodiment: registry, adapter boundary, IK, and one recorded CaP episode.

Needs the insertion_skill package and its generated scenes (``envs/<task>/``); skipped
otherwise. Rendering uses EGL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("INSERTION_HOST_EVIDENCE", "1")
pytest.importorskip("mujoco")
pytest.importorskip("insertion_skill")

from cap_harness.contracts import ArmCommand, MotionStrategy, RobotAction  # noqa: E402
from cap_harness.insertion.adapter import InsertionAdapter  # noqa: E402
from cap_harness.insertion.registry import (  # noqa: E402
    InsertionRegistryError,
    InsertionTaskRegistry,
)
from cap_harness.registry import INSERTION_PUBLIC_TOOL_NAMES  # noqa: E402

REGISTRY = InsertionTaskRegistry()
TASK = "usb_c_into_receptacle"
pytestmark = pytest.mark.skipif(
    TASK not in REGISTRY.available_tasks, reason="insertion scenes have not been generated"
)


@pytest.fixture
def adapter():
    instance = InsertionAdapter(camera_width=96, camera_height=72)
    yield instance
    instance.close()


def test_registry_resolves_task_refs():
    metadata = REGISTRY.resolve(f"{TASK}:0")
    assert metadata.task_name == TASK
    assert REGISTRY.resolve((TASK, 0)) is metadata
    with pytest.raises(InsertionRegistryError):
        REGISTRY.resolve(TASK, 1)
    with pytest.raises(InsertionRegistryError):
        REGISTRY.resolve("no_such_task")


def test_seed_zero_starts_on_the_nominal_standoff(adapter):
    adapter.reset(TASK, 0)
    record = adapter.episode_record
    assert record["perturbation_scale"] == 0.0
    assert record["start_penetrations"] == []
    tip = np.asarray(adapter.tip_error_in_target())
    metadata = adapter.get_task_metadata()
    assert np.hypot(tip[0], tip[1]) < 1e-4
    assert tip[2] == pytest.approx(metadata["seated_depth_m"] + metadata["standoff_m"], abs=2e-4)


def test_perturbed_seeds_are_reproducible_and_contact_free(adapter):
    adapter.reset(TASK, 7)
    first = dict(adapter.episode_record)
    adapter.reset(TASK, 7)
    assert dict(adapter.episode_record) == first
    assert first["start_penetrations"] == []
    assert first["perturbation_scale"] == 1.0


def test_metadata_hides_ground_truth(adapter):
    adapter.reset(TASK, 3)
    metadata = adapter.get_task_metadata()
    keys = " ".join(metadata).lower()
    assert "success" not in keys.replace("success_tolerance_m", "")
    assert "seated_tip_pose_estimate" in metadata
    assert not any(name.split(".", 1)[1].startswith("tip_error") for name in INSERTION_PUBLIC_TOOL_NAMES)


def test_step_holds_one_control_period_and_ik_round_trips(adapter):
    adapter.reset(TASK, 0)
    state = adapter.get_robot_state()
    solved = adapter.kinematics().solve_ik(state.end_effector_poses["primary"], state)
    assert solved.ok
    assert np.allclose(solved.joint_positions, state.joint_positions["primary"], atol=1e-6)
    result = adapter.step(
        RobotAction({"primary": ArmCommand("joint_position", solved.joint_positions, embodiment="insertion")})
    )
    assert result.ok
    assert result.diagnostics["step_index"] == 1
    assert set(result.observation.cameras) == {"closeup", "overview"}


def test_gripper_is_locked(adapter):
    adapter.reset(TASK, 0)
    result = adapter.set_gripper(1.0)
    assert not result.ok
    assert result.error.code.value == "unsupported"


def test_recorded_episode_writes_host_only_evaluation(tmp_path: Path):
    from cap_harness.run import run_program

    program = tmp_path / "hold.py"
    program.write_text(
        "md = insertion.get_task_metadata()\n"
        "state = get_robot_state()\n"
        "held = move_to_joints(list(state.joint_positions['primary']), max_steps=3)\n"
        "result = {'depth': md['seated_depth_m'], 'held': held.ok}\n",
        encoding="utf-8",
    )
    outcome = run_program(
        benchmark="insertion",
        suite=TASK,
        task_id=0,
        seed=0,
        program_path=program,
        output_root=tmp_path / "out",
        camera_width=96,
        camera_height=72,
        capture_videos=False,
    )
    assert outcome.program_ok
    assert outcome.task_success is False
    evaluation = json.loads((outcome.run_dir / "evaluation" / "insertion.json").read_text())
    assert evaluation["task_success"] is False
    assert evaluation["steps"] >= 1
    assert MotionStrategy(ik_solver="mujoco").ik_solver == "mujoco"


def _press(adapter, depth_past_mouth_m, ticks=30):
    """Command the nominal tip straight down the socket estimate's axis; return the last wrench."""
    from cap_harness.contracts import Pose
    from cap_harness.geometry import matrix_to_pose

    metadata = adapter.get_task_metadata()
    estimate = metadata["seated_tip_pose_estimate"]
    target = Pose(
        np.array(estimate["position"]), np.array(estimate["quaternion_wxyz"]), frame=estimate["frame"]
    ).as_matrix().copy()
    target[:3, 3] += target[:3, 2] * (metadata["seated_depth_m"] - depth_past_mouth_m)
    state = adapter.get_robot_state()
    solved = adapter.kinematics().solve_ik(matrix_to_pose(target, frame=estimate["frame"]), state)
    assert solved.ok
    start = state.joint_positions["primary"]
    for tick in range(ticks):
        joints = start + min(1.0, 2.0 * (tick + 1) / ticks) * (solved.joint_positions - start)
        result = adapter.step(
            RobotAction({"primary": ArmCommand("joint_position", joints, embodiment="insertion")})
        )
        assert result.ok
    return np.asarray(result.diagnostics["wrench_tip"])


def test_force_is_off_by_default(adapter):
    adapter.reset(TASK, 0)
    assert not adapter.force_enabled
    assert adapter.get_wrench is None
    assert "force_torque" not in adapter.get_controller_metadata()
    assert adapter.get_task_metadata()["sensing"].startswith("no force/torque")
    state = adapter.get_robot_state()
    result = adapter.step(
        RobotAction({"primary": ArmCommand("joint_position", state.joint_positions["primary"], embodiment="insertion")})
    )
    assert "wrench_tip" not in result.diagnostics
    with pytest.raises(RuntimeError):
        adapter.wrench()


def test_force_reads_zero_in_free_space_and_contact_when_seated(monkeypatch):
    monkeypatch.setenv("INSERTION_FORCE", "1")
    instance = InsertionAdapter(camera_width=64, camera_height=48)
    try:
        instance.reset(TASK, 0)
        assert instance.force_enabled
        assert "force_torque" in instance.get_controller_metadata()
        assert np.allclose(instance.wrench(), 0.0, atol=0.05)
        # Free motion (and the small wrist rotation it needs) stays near zero: the tare holds.
        free = _press(instance, -0.004)
        assert np.linalg.norm(free[:3]) < 0.1
        # Driving 2 mm past the seat bottoms the plug out: the socket pushes back along +z.
        seated = _press(instance, instance.get_task_metadata()["seated_depth_m"] + 0.002, ticks=60)
        assert instance.check_success()
        assert seated[2] > 2.0
        reading = instance.get_wrench()
        assert reading["frame"] == "tip"
        assert reading["force_n"] == pytest.approx(seated[:3].tolist())
    finally:
        instance.close()


def test_force_noise_is_seeded_and_does_not_change_the_episode(monkeypatch):
    monkeypatch.setenv("INSERTION_FORCE", "1")
    monkeypatch.setenv("INSERTION_FT_NOISE_N", "0.3")
    monkeypatch.setenv("INSERTION_FT_BIAS_N", "1.0")
    monkeypatch.setenv("INSERTION_FT_LOWPASS_HZ", "30")
    readings = []
    for _ in range(2):
        instance = InsertionAdapter(camera_width=64, camera_height=48)
        try:
            instance.reset(TASK, 5)
            record = dict(instance.episode_record)
            readings.append(_press(instance, -0.004, ticks=10))
        finally:
            instance.close()
    assert np.array_equal(readings[0], readings[1])
    assert np.linalg.norm(readings[0][:3]) < 0.5
    monkeypatch.delenv("INSERTION_FORCE")
    plain = InsertionAdapter(camera_width=64, camera_height=48)
    try:
        plain.reset(TASK, 5)
        assert dict(plain.episode_record) == record
    finally:
        plain.close()


@pytest.mark.parametrize("force", [False, True])
def test_get_wrench_tool_is_registered_only_with_force(tmp_path: Path, monkeypatch, force):
    from cap_harness.run import run_program

    if force:
        monkeypatch.setenv("INSERTION_FORCE", "1")
    program = tmp_path / "wrench.py"
    program.write_text(
        "w = insertion.get_wrench()\n"
        "assert len(w['force_n']) == 3 and abs(w['force_n'][2]) < 0.1\n"
        "result = {'fz': w['force_n'][2]}\n",
        encoding="utf-8",
    )
    outcome = run_program(
        benchmark="insertion",
        suite=TASK,
        task_id=0,
        seed=0,
        program_path=program,
        output_root=tmp_path / "out",
        camera_width=64,
        camera_height=48,
        capture_videos=False,
    )
    assert outcome.program_ok is force


def _vision_error_m(adapter, metadata):
    """True lateral/axial error (m) of the reported seated-tip estimate, in the target frame."""
    from cap_harness.contracts import Pose
    from cap_harness.geometry import pose_to_matrix

    env = adapter.native_env
    world_from_target = np.eye(4)
    world_from_target[:3, :3] = env.data.site_xmat[env.target].reshape(3, 3)
    world_from_target[:3, 3] = env.data.site_xpos[env.target]
    record = metadata["seated_tip_pose_estimate"]
    base_from_estimate = pose_to_matrix(
        Pose(position=record["position"], quaternion_wxyz=record["quaternion_wxyz"], frame=record["frame"])
    )
    error = np.linalg.inv(world_from_target) @ adapter._world_from_base @ base_from_estimate
    return float(np.hypot(error[0, 3], error[1, 3])), float(error[2, 3])


def test_vision_socket_estimate_geometry_only_without_detector(adapter, monkeypatch):
    """INSERTION_SOCKET_SOURCE=vision with the detection service unreachable: the depth geometry
    still finds the opening from the prior ROI, and the reported bounds widen (no detection)."""
    monkeypatch.setenv("INSERTION_SOCKET_SOURCE", "vision")
    monkeypatch.setenv("INSERTION_VISION_URL", "http://127.0.0.1:9")
    adapter.reset(TASK, 3)
    metadata = adapter.get_task_metadata()
    assert metadata["socket_estimate_source"]["source"] == "vision"
    assert metadata["socket_estimate_source"]["status"] in ("geometry_only", "weak_prior", "fallback_prior")
    lateral, axial = _vision_error_m(adapter, metadata)
    bounds = metadata["uncertainty"]
    assert lateral <= bounds["socket_estimate_lateral_m"]
    assert abs(axial) <= bounds["socket_estimate_axial_m"]
    assert adapter.get_task_metadata() is not metadata  # fresh mapping, cached estimate
    assert adapter.episode_record["socket_vision"]["status"] == metadata["socket_estimate_source"]["status"]


def test_vision_socket_estimate_with_detector(adapter, monkeypatch):
    """Full pipeline (Grounding DINO / OWLv2 + SAM 2 service); skipped when it is not running."""
    import urllib.request

    from cap_harness.insertion.perception import VisionClient

    try:
        urllib.request.urlopen(VisionClient().url + "/health", timeout=2).read()
    except Exception:
        pytest.skip("vision service not running (perception/start_server.sh)")
    monkeypatch.setenv("INSERTION_SOCKET_SOURCE", "vision")
    adapter.reset(TASK, 3)
    metadata = adapter.get_task_metadata()
    assert metadata["socket_estimate_source"]["status"] == "ok"
    lateral, axial = _vision_error_m(adapter, metadata)
    assert lateral < 0.3e-3 and abs(axial) < 0.2e-3
    assert lateral <= metadata["uncertainty"]["socket_estimate_lateral_m"]
