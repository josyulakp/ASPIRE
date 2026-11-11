# Insertion (PR V4P1SR + chopsticks)

The `insertion` benchmark runs the eight connector-insertion scenes from the sibling
`insertion_skill` package through the CaP harness. At t = 0 the robot already holds the plug
above the socket. The task is the last centimetre: find the opening and seat the plug.

By default there is **no force/torque sensing**. The program sees proprioception, a perception-grade
socket estimate, and RGB-D. Contact shows up only as joint tracking error. `INSERTION_FORCE=1` adds an
opt-in wrist wrench; see [Wrist force/torque](#wrist-forcetorque-opt-in).

| suite (task_id 0) | plug → socket |
|---|---|
| `adapter_into_wall_socket` | Apple 20 W adapter → Indian 6 A wall socket |
| `nvme_into_m2_dock` | M.2 2280 SSD → USB-C dock (vertical) |
| `nvme_into_m2_socket` | M.2 2280 SSD → board M.2 socket (horizontal, side grasp) |
| `pcie_card_into_x16_slot` | quad-M.2 PCIe card → x16 slot |
| `pcie_riser_into_x16_slot` | riser plug → x16 slot |
| `ram_into_dimm_slot` | DDR4 DIMM → DIMM slot |
| `usb_a_into_receptacle` | USB-A plug → front-panel receptacle |
| `usb_c_into_receptacle` | USB-C plug → front-panel receptacle |

## Setup

`insertion_skill` needs `mujoco>=3.13`, while ASPIRE's uv override pins `mujoco==3.3.0`, so use a separate venv that
doesn't apply the override. Run this from outside the ASPIRE directory, so uv doesn't pick up its project settings:

```bash
uv venv --python 3.12 ../.venv-aspire
VIRTUAL_ENV=../.venv-aspire uv pip install -e ./ASPIRE -e "..[render]" \
    "mujoco==3.13.0" "numpy==1.26.4" "scipy==1.11.4" "imageio[ffmpeg]" pytest
export INSERTION_ASSETS=/path/to/insertion_assets   # task meshes; needed to (re)build scenes
python -m insertion_skill.build_envs                 # scene.xml holds absolute paths: build per machine
python tools/validate_envs.py                        # every task: start ok, hold, F/T, seated, baseline insert
```

- `numpy` 2.x breaks `insertion_skill.env.InsertionEnv.gravity()`, so keep 1.26.
- Rendering uses EGL: `export MUJOCO_GL=egl`.

## Running

```bash
cap-harness run --benchmark insertion --suite usb_c_into_receptacle --task-id 0 --seed 3 \
    --program my_program.py --camera-width 480 --camera-height 360
cap-harness run-batch --benchmark insertion --suite ram_into_dimm_slot --task-id 0 \
    --seeds 1 2 3 --program my_program.py
```

- The episode horizon is 1200 ticks (60 s at 20 Hz).
- Seed 0 is the unperturbed scene. Every other seed draws the hidden errors described below.

| variable | default | effect |
|---|---|---|
| `INSERTION_ENVS` | `<insertion_skill>/envs` | where the generated scenes are found |
| `INSERTION_PERTURBATION_SCALE` | `1.0` | multiplies every hidden error (curriculum) |
| `INSERTION_PERTURBATION` | `full` | `translation` or `rotation` to draw only those error components |
| `INSERTION_CAMERA_REFRESH_TICKS` | `1` | re-render the cameras every N ticks (speed) |
| `CAP_HARNESS_STEP_KEYFRAMES` | `1` | `0` skips the before/after keyframe PNGs on `step()` calls |
| `INSERTION_FORCE` | `0` | `1` enables the wrist wrench (`insertion.get_wrench()`, `adapter.wrench()`) |
| `INSERTION_FT_NOISE_N` | `0` | white noise std per 2 kHz sensor sample, N (torque: × 0.02 m) |
| `INSERTION_FT_BIAS_N` | `0` | bias drift after the tare, N per minute, random direction per seed (torque: × 0.02 m) |
| `INSERTION_FT_LOWPASS_HZ` | `0` | first-order low-pass cutoff; `0` reports the mean over each control period |
| `INSERTION_SOCKET_SOURCE` | `synthetic` | `vision`: the socket estimate and its `socket_estimate_*` uncertainty come from the camera pipeline (see below) |
| `INSERTION_VISION_URL` | `http://127.0.0.1:8765` | the detection service (`insertion_skill/perception/start_server.sh`) |

## What a program sees

- **Arm.** One arm, `primary`, with 6 joints (`pr_right_joint1..4, 6, 7`; joint 5 is locked).
- **Control.** Joint-position targets, held for one control period by a joint impedance controller: gravity
  compensation plus PD. Stiffness is in `insertion.get_controller_metadata()`.
  - Wrist joint 6 is stiff enough to slide a lightly pressed plug across a face.
  - Wrist roll (joint 7) is soft. Its axis is the insertion axis for every axial grasp.
- **Gripper.** Locked, because the plug is pre-grasped. `set_gripper` returns `unsupported`.
- **End-effector pose.** `end_effector_poses["primary"]` is the plug tip *as the nominal grasp places it*, in
  `pr_base_link`. Its +z points out of the socket.
- **Metadata.** `insertion.get_task_metadata()` returns:
  - `seated_tip_pose_estimate` and `socket_mouth_pose_estimate`;
  - `insertion_direction`, `seated_depth_m`, `standoff_m`;
  - `success_tolerance_m` and `plug_bounds_in_tip_frame_m`;
  - the grasp type and jaw gap;
  - `uncertainty`: bounds of the hidden errors.
- **IK.** `solve_ik` and `move_to_pose` use the in-process `mujoco` IK backend, a damped least-squares solve on the
  nominal tip frame.
- **Cameras.** `closeup` looks at the socket from 40° off-axis. `overview` is the scene camera.
- **No perception services.** There are none, so `localize_object` and `generate_grasps` aren't usable here.

**Vision socket estimate** (`INSERTION_SOCKET_SOURCE=vision`, `src/cap_harness/insertion/perception.py`).
On the first `get_task_metadata()` of an episode the adapter runs one socket scan and caches it:

- Four RGB-D look views (1280 × 960) aimed at the *prior* mouth (the start tip pose minus the standoff), with the
  arm and plug out of view, like a station camera imaging the socket before the arm moves in. View 0 is 5° off the
  prior axis along the plug's long axis; three oblique views sit at 35°.
- Grounding DINO + OWLv2 (task prompts) gated by the projected prior, and a SAM 2 mask, from a GPU service
  (`insertion_skill/perception/vision_server.py`). Without the service the depth geometry still runs on the prior
  ROI, and the bounds widen.
- Depth geometry: face planes facing the prior axis; rays that pass more than 0.8 mm behind a plane mark its aperture;
  enclosed openings, key-split slots merged, two-pin pairs; the slot key from the near-axis view; yaw from the
  opening axis. Seat = mouth − `seated_depth` along the face normal, or the floor seen through a lipped slot.
- `uncertainty.socket_estimate_*` are the measured bounds: a per-regime floor fitted on calibration seeds 101–130
  plus the leave-one-view-out spread, ×2 when the views disagree, the prior's bounds on failure.
  `socket_estimate_source` gives the status and the time taken.

**Hidden per-seed errors** (scale 1):

- **Plug in the jaws:** ≤1 mm lateral, ±0.5 mm axial, ±1° tilt, ±1° yaw.
- **Socket estimate:** ≤1 mm lateral, ±0.5 mm axial, ±0.5° tilt, ±1° yaw.
- **Start pose:** ≤2 mm lateral, up to +3 mm higher, ±1°. If the plug would touch the socket at the start (the
  DIMM's tall guides), it backs off along the axis.

**Success.** `check_success()` is `InsertionEnv.seated()`: the true plug tip is within 0.5 mm axially and 1 mm
laterally of the seated target.

With `INSERTION_HOST_EVIDENCE=1` the adapter writes host-only evidence to `<run>/evaluation/insertion.json` (and the hidden perturbations into `episode/reset.json`). It is off by default, because run artifacts are what a debugging agent reads. It contains the true final
tip error, a per-tick true tip trace, and the hidden perturbations. The program never sees any of it.

## Wrist force/torque (opt-in)

With `INSERTION_FORCE=1` the adapter reads the wrist F/T sensor (`pr_fts`, sampled at the 2 kHz simulator rate)
and reports the **contact wrench the environment applies to the held plug**:

- **Compensated and tared.** The predicted gravity load of the hand and the plug is subtracted at every sample, so
  reorienting the wrist doesn't read as contact. The rest is tared at reset, after the arm settles (mean of 0.1 s).
  Free motion reads about 0.01 N.
- **Frame.** The wrench is moved to the nominal plug tip and expressed in the nominal tip frame, the frame of
  `end_effector_poses["primary"]`. Its +z points out of the socket, so pressing the plug into a face or bottoming
  it out reads **+Fz**.
- **Rate.** The reading is latched once per control period (20 Hz). It's the mean over the period's 100 samples,
  or the output of a first-order low-pass when `INSERTION_FT_LOWPASS_HZ` is set.
- **Realism knobs.** `INSERTION_FT_NOISE_N` and `INSERTION_FT_BIAS_N` (see the table above) are seeded from the
  episode seed on a separate random stream. They never change the seed's hidden perturbations.

How to read it:

- **CaP programs** call `insertion.get_wrench()`. It returns `{"force_n": [3], "torque_nm": [3], "frame": "tip",
  "timestamp_s"}`. The tool is registered only when force is enabled, and `get_task_metadata()["sensing"]` says
  which mode is active.
- **Non-CaP clients** (an RL loop) call `adapter.wrench()`, which returns a `(6,)` array
  `[Fx, Fy, Fz, Tx, Ty, Tz]`. The same values are in `step(...).diagnostics["wrench_tip"]`.
  `get_controller_metadata()["force_torque"]` describes the frame, rate, filter and noise settings.
- **Disabled.** Without `INSERTION_FORCE=1`, nothing changes: `adapter.get_wrench` is `None` and `adapter.wrench()`
  raises.

Scripted-press check at seed 0, clean sensor (mean of the last 10 ticks of each phase):

| phase | USB-C Fz / \|Fxy\| (N) | DIMM Fz / \|Fxy\| (N) |
|---|---|---|
| hold and free motion at standoff (±3 mm lateral) | 0.00 / 0.01 | 0.00 / 0.00 |
| pressed 3 mm past the mouth, 2.5 mm off axis | 14.2 / 3.8 (face) | 0.3 / 1.1 (guide side load) |
| seated | 0.02 / 0.01 | 0.01 / 0.00 |
| driven 2 mm past seated (bottom-out) | 10.0 / 2.6 | 5.4 / 1.5 |

With noise 0.3 N, drift 1 N/min and a 30 Hz low-pass, the free-motion std of \|F\| is 0.06 N. After 15 s the drift
has reached about 0.2 N.

## Skill library

Insertion skills live in
[`../.claude/skills/iterative-debugging/skills/insertion.md`](../.claude/skills/iterative-debugging/skills/insertion.md)
and grow through the iterative-debugging campaign.

The generic program from the 2026-10-20 campaign is
[`../examples/insertion/generic_insertion.py`](../examples/insertion/generic_insertion.py). It is one program for
all 8 tasks, written from the library alone, and scored 373/400 on held-out seeds 1–50. Debugging agents must not
read it.
