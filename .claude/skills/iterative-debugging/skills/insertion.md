---
name: insertion
description: Structural template for seating a pre-grasped connector into a socket on the `insertion` benchmark (no force/torque sensing). Covers the frames, metadata and per-tick command loop that every insertion program needs. No task-specific or search strategies yet; those are discovered through experiment and added as their own subsections (trigger → code → evidence).
---

# Insertion: code template

> This file provides only the **structural skeleton**. Strategies for approach, contact detection,
> search and seating are discovered through experiment and added here once a verifier has reproduced them.
> Programs can't import modules (other than `math`), so every helper below is plain Python.

## What the embodiment gives you

- **The plug is already held** above the socket, and the gripper is locked.
- **`insertion.get_task_metadata()`** returns `seated_tip_pose_estimate` (position, `quaternion_wxyz`, frame), whose
  +z points out of the socket, plus `seated_depth_m`, `standoff_m`, `insertion_direction`, `success_tolerance_m`,
  `plug_bounds_in_tip_frame_m`, the grasp type and jaw gap, and `uncertainty` (bounds of the hidden errors).
- **The end-effector pose** (`get_robot_state().end_effector_poses["primary"]`) is the plug tip *as the nominal
  grasp places it*, in `pr_base_link`.
- **There is no force reading.**
- **Control** runs at 20 Hz (`insertion.get_controller_metadata()["control_frequency_hz"]`), so one `step` is
  50 ms. The episode budget is in `get_task_context().metadata["episode_steps_remaining"]`.

## Frame helpers and a one-tick command

Work in the estimate frame: local = R_estᵀ (p − p_est).

```python
import math
md = insertion.get_task_metadata()
seat = md["seated_tip_pose_estimate"]
FRAME = seat["frame"]
P0 = list(seat["position"])
Q0 = list(seat["quaternion_wxyz"])

def qmul(a, b):
    return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]

def qrot(q, v):
    p = qmul(qmul(q, [0.0, v[0], v[1], v[2]]), [q[0], -q[1], -q[2], -q[3]])
    return [p[1], p[2], p[3]]

EX, EY, EZ = qrot(Q0, [1, 0, 0]), qrot(Q0, [0, 1, 0]), qrot(Q0, [0, 0, 1])

def to_base(local):
    return [P0[i] + EX[i]*local[0] + EY[i]*local[1] + EZ[i]*local[2] for i in range(3)]

def tip_local():
    p = get_robot_state().end_effector_poses["primary"].position
    d = [float(p[i]) - P0[i] for i in range(3)]
    return [sum(d[i]*axis[i] for i in range(3)) for axis in (EX, EY, EZ)]

def command(local):
    # One control tick toward a tip position in the estimate frame (orientation Q0).
    ik = solve_ik(Pose(position=to_base(local), quaternion_wxyz=Q0, frame=FRAME))
    if not ik.ok:
        return False
    q = [float(v) for v in ik.joint_positions]
    return step(RobotAction({"primary": ArmCommand("joint_position", q, embodiment="insertion")})).ok
```

Each `command` costs exactly one tick. `move_to_pose` plans and executes many ticks, and returns `ok=False` when
the final waypoint doesn't converge.

## Sandbox notes

- Loop variables must not be named `_`, because private names are rejected.
- `global` and `nonlocal` aren't available, so keep mutable state in a list or dict.
- `ArmCommand` needs `embodiment="insertion"`, or it is validated as a 7-joint arm.

## Routing: one program for all eight connectors

The phase-4 generalist composed the entries below into ONE program, using only this library. It routes only on
plug bounds, metadata and what it measures, never on the task name.

| development 101–110 | unseen 111–115 | held-out 1–50 |
|---|---|---|
| 78/80 | 38/40 | **373/400** |

Held-out per task:

| task | held-out |
|---|---|
| adapter | 50 |
| M.2 dock | 50 |
| PCIe card | 50 |
| riser | 49 |
| USB-A | 48 |
| USB-C | 46 |
| DIMM | 43 |
| M.2 board socket | 37 |

The routing:

1. **Align and descend.** Align laterally at the start height, then descend slowly.
   - A stall far above the mouth means tall guide posts: run "RGB-D card/slot alignment", then "Tall guide
     posts".
   - Otherwise press as in "First try", and accept a seat only through "Hard-stop seat test" (with the lateral
     check).
2. **If the press stalls, choose the search from the plug bounds** (`EXT` = the plug's x/y extents, `THIN` = the
   index of the smaller one):
   - **Compact plug** (`EXT[1-THIN] < 4 * EXT[THIN]`): "Peck probe with drift calibration".
   - **Long edge card** (`0.5 * EXT[1-THIN] * (grasp_yaw_rad + socket_estimate_yaw_rad) > EXT[THIN]`, so its yaw
     error alone exceeds its thickness at the ends):
     1. correct yaw, lean and thin-axis offset with "RGB-D card/slot alignment";
     2. thin-axis scan;
     3. "Edge card sits on the slot dip floor" (loaded wide pecks);
     4. "Edge card stalls on the housing top" step 2 (forceful re-presses) as the fallback.
   - **Short edge card** (neither of the above): "Edge card in a recessed connector", the 2-D grid, with the face
     taken from the first press and every seat gated by the hard-stop test.

**Why split cards by length, not by scan profile.** Scan-first routing misread the M.2 cards: their scans show
yielding dips beside the socket and shelf dips, not slot dips. The long/short split comes from the bounds alone, at
zero ticks.

**Limits.**
- **The routing thresholds are one agent's choice.** They were validated on these 8 connectors.
- **Weakest regimes:**
  - the M.2 board socket, where a flat face with a shallow centre dip exhausts the grid;
  - the DIMM, where a few seeds exhaust the tall-guide retries;
  - compact grids that run out on USB-C/USB-A (6 held-out seeds).

## First try: align above the mouth, then a slow straight press past the seat

**Trigger.** Run this first on any insertion task, before any search: the plug is held over the socket and no
stall has been observed yet.
- **If it seats:** you are done. Self-aligning connectors (the round-pin wall adapter) absorbed up to ~1.4 mm of
  lateral error through compliance.
- **If the press ends with the tip clearly above the seat** (`tip_local()[2]` above ~0.5 mm, where 0 is the seat
  estimate): it stalled on the face or partly inside it, so go to a search entry below. On the M.2 dock, stalls
  ranged from 0.6 to 3.4 mm, so a threshold tied to the mouth (`seated_depth_m - 1 mm`) misses the shallow ones. A straight press like this one seated only 3–4 of 10 seeds on the PCIe card and on
  USB-C.

**Code.** It uses `command()` and `tip_local()` from the template; local z = 0 is the seat estimate.

- Tick cost: 40 ticks to align, plus (`seated_depth_m` + 2 mm − Z_END) / 0.5 mm ticks to press, plus the dwell.
  That came to 120 ticks on the adapter.
- The plug stays held throughout.

```python
unc = md["uncertainty"]
DEPTH = float(md["seated_depth_m"])
# Over-press by the axial-uncertainty sum: the true seat can sit that much deeper than the estimate.
Z_END = -(float(unc["grasp_axial_m"]) + float(unc["socket_estimate_axial_m"]))
RATE = 0.0005  # 0.5 mm/tick; 2 mm/tick jammed at the mouth
z_hi = DEPTH + 0.002
start = tip_local()
for i in range(30):  # align laterally on the way to 2 mm above the mouth
    a = (i + 1) / 30.0
    command([start[0] * (1 - a), start[1] * (1 - a), start[2] * (1 - a) + z_hi * a])
for k in range(10):
    command([0.0, 0.0, z_hi])
z = z_hi
while z > Z_END:
    z = max(Z_END, z - RATE)
    command([0.0, 0.0, z])
for k in range(40):  # dwell; 5 ticks also passed
    command([0.0, 0.0, Z_END])
```

**Evidence.**

| run | development | unseen | held-out |
|---|---|---|---|
| proposer (adapter_into_wall_socket) | 10/10 | 5/5 | 50/50 |
| verifier, this snippet applied as written | 5/5 (120 ticks) | – | – |

The proposer's one-variable ablations:

| change | development |
|---|---|
| no over-press | 8/10 (true seat ~0.8 mm deeper) |
| 2 mm/tick press | 9/10 (jammed at the mouth) |
| no pre-alignment | 9/10 (jammed at the mouth) |

These are one program's ablations, not independent evidence.

**Routing evidence.** On nvme_into_m2_dock this press scored 3/10. The rest stalled at z 0.6–3.4 mm, which fired
the stall trigger and correctly sent the program to a search. The dock graded this entry `useful` for that
routing.

**Limits.**
- **Stall retry.** A back-off-and-re-press stall retry was proposed, but the verifier never triggered it, so it is
  untested and not included here.
- **Keep any search press shallow.** On the M.2 dock, commanding 5–8 mm past the contact deflected the measured
  tip 2–3 mm sideways. This is one program's measurement. Searches that worked capped the command at 1.5–4 mm
  below the measured tip.

## Edge card stalls on the housing top: peck-scan the thin axis, then re-press

**Trigger.** A straight push along the estimated axis stalls within ~1 mm of the mouth: `tip_local()[2]` stays
above `seated_depth_m - 1 mm` while you command deeper. The plug is a long edge card: its wide extent is many times
its thin extent in `plug_bounds_in_tip_frame_m`. Validated on the PCIe x16 card. For compact plugs, see "Peck probe
with drift calibration" below.

**Code.** It uses `command()` and `tip_local()` from the template.

- The scan is 17 pecks of 19 ticks each (323 ticks at SPAN 2.97 mm).
- Each re-press costs 16 ticks, plus 40 to press and 8 to lift for each entry that does not end seated. The worst
  case is 7 + 8 presses.
- The plug stays held in the jaws throughout.

```python
bmin = md['plug_bounds_in_tip_frame_m']['min']
bmax = md['plug_bounds_in_tip_frame_m']['max']
THIN = 0 if (bmax[0] - bmin[0]) < (bmax[1] - bmin[1]) else 1
unc = md['uncertainty']
DEPTH = float(md['seated_depth_m'])
SPAN = float(unc['grasp_lateral_m']) + float(unc['socket_estimate_lateral_m']) + 0.5 * float(unc['grasp_tilt_rad']) * (bmax[2] - bmin[2])
PITCH = 0.00035
z_hi = DEPTH + 0.0025
wide = [0.0]

def lat(t, z):
    # t on the thin lateral axis, wide[0] on the other one, z on local z
    local = [0.0, 0.0, z]
    local[THIN] = t
    local[1 - THIN] = wide[0]
    return local

# 1. Peck scan across the thin axis: lift, press 5 mm past the seat, read z, lift.
n_side = int(SPAN / PITCH)
prof = []
for t in [PITCH * (i - n_side) for i in range(2 * n_side + 1)]:
    for k in range(4): command(lat(t, z_hi))
    for k in range(12): command(lat(t, DEPTH - 0.005))
    prof.append([t, tip_local()[2]])
    for k in range(3): command(lat(t, z_hi))
top = max([p[1] for p in prof])
housing = [i for i in range(len(prof)) if prof[i][1] > top - 0.0004]
# Only points BETWEEN the outermost housing contacts can be the slot. Points outside
# that drop deep are the plug falling off the housing edge, which looks like seating in z.
inner = [prof[i] for i in range(min(housing) + 1, max(housing)) if prof[i][1] <= top - 0.0004]
deep = [p for p in inner if p[1] < DEPTH - 0.0035]
dips = [p for p in inner if p[1] >= DEPTH - 0.0035]
group = deep if deep else dips
entered = None
done = False
if group:
    # 2. Re-press inside the dip: fine thin offsets, then wide-axis offsets out to +-2 mm.
    t_lo = min([p[0] for p in group]) - PITCH / 2
    t_hi = max([p[0] for p in group]) + PITCH / 2
    dip = min([p[1] for p in group])
    mid = 0.5 * (t_lo + t_hi)
    fine = [mid] + [mid + s * k * PITCH / 3 for k in range(1, 8) for s in (1, -1) if t_lo <= mid + s * k * PITCH / 3 <= t_hi]
    WIDE = (0.0005, -0.0005, 0.001, -0.001, 0.0015, -0.0015, 0.002, -0.002)
    tries = [[t, 0.0] for t in fine] + [[mid, w] for w in WIDE]
    for tw in tries:
        if get_task_context().metadata["episode_steps_remaining"] < 150:
            break
        wide[0] = tw[1]
        for k in range(4): command(lat(tw[0], z_hi))
        for k in range(12): command(lat(tw[0], DEPTH - 0.008))
        if tip_local()[2] < min(dip, top - 0.0004) - 0.001:
            entered = tw
            z0 = tip_local()[2]  # press on to 2 mm past the seat estimate
            for k in range(20): command(lat(tw[0], z0 + (-0.002 - z0) * (k + 1) / 20.0))
            for k in range(20): command(lat(tw[0], -0.002))
            cmd = lat(tw[0], 0.0)
            m = tip_local()
            # A wedged card can stop near seat height with the arm deflected 6-7 mm sideways.
            done = m[2] < 0.0005 and abs(m[0] - cmd[0]) < 0.002 and abs(m[1] - cmd[1]) < 0.002
            if done:
                break
            for k in range(8): command(lat(tw[0], z_hi))  # not seated: lift out and keep searching
    wide[0] = 0.0
# done is False: no try seated (entered is None: nothing entered at all).
```

**Why it works.** On the card, misalignment along the thin axis puts the plug on the housing top, where the slot
shows up as a 0.35–1.5 mm dip between housing contacts. Once the plug is over the dip, the remaining jam comes
from the offset along the slot's long axis (the key), which the wide-axis presses search.

**Evidence.**

| run | development seeds | unseen | v00 |
|---|---|---|---|
| proposer (pcie_card_into_x16_slot) | 10/10 | 4/5 | 3/10 |
| verifier, from this text alone | 4/5 | – | – |

On the verifier's seeds, the wide-axis presses made 3 of the 4 entries and the fine thin presses made 1.

**Limits.**
- **Wide offsets extend to ±2 mm.** The phase-3 rerun's edit was verified 5/5, with one seed entering only at
  +2.0 mm. Under load the pressed tip drifts about −1.6 mm along the wide axis.
- **Two branches were never exercised in verification:** lift-and-continue, and the lateral seat check rejecting a
  seat. True seats deviated 1.4–1.6 mm laterally against the 2 mm limit, so the margin is only 0.4–0.6 mm.
- **Wedge jam.** A card whose dip is already deeper than 3.5 mm can wedge under the forceful re-press. It was
  unsolved in the rerun; one probe freed it with a +1.5° yaw.
- **Shallow dips are missed.** A dip shallower than the 0.4 mm cutoff isn't recognised; the proposer's one unseen
  failure was a 0.35 mm dip.
- **Moving the command toward the measured tip while pressing hurt one program.** The proposer measured that
  pressing shifts the measured tip by 1–1.5 mm laterally. A version that moved the command toward the measured tip
  while pressing scored 0/10. That is one program's ablation, not an independent result.
- **Precondition: the re-press must go deeper.** This failed on pcie_riser_into_x16_slot, which graded the entry
  `misleading`. There the step-2 re-presses from `z_hi`, including wide offsets of ±1.5 mm, all stayed on the dip
  floor (~1.4 mm below the housing), and the entry applied as written scored 0/10.
  - The runnable check: if every entry in `tries` ends with `tip_local()[2] > dip - 0.001`, so `entered` is
    still `None`, this step-2 recipe does not apply to the connector. Lifting and re-pressing will not fix it.
  - On the riser, the plug caught only when the wide-axis search was done *while pressed* 1.5 mm past the dip
    floor, continuing from the catch without lifting. Use the next entry, "Edge card sits on the slot dip floor".
    It was verified on the riser and transferred to this card at 4/5.
  - Both tasks agree that the final press should go past the seat estimate (to local z −2…−3 mm). The slot floor
    is the true seat.
- **Precondition: the scan must find an inner dip.** This failed on nvme_into_m2_dock, which graded the entry
  `misleading`. The M.2 card matches the "thin edge card" shape, but in a recessed dock the scan saw one flat face
  plateau with drops only at its edges. Those drops were a shelf (face − 1.75 mm) and a cavity (face − 5..7 mm)
  beside the connector, not the slot. As written, the entry scored 3/10, no better than a straight press.
  - The runnable check: if `group` is empty, this entry does not apply.
  - In that case use a 2-D lift-move-press grid with a face-relative seat test. The structure of the compact-plug
    entry below solved the dock at 10/10. A dock-specific entry is under verification.
  - On the dock, the absolute entry threshold here also rejected presses that had reached the true seat.

## Peck probe with drift calibration, then a ring-ordered grid (compact plugs)

**Trigger.** A straight press at the socket estimate stalls at the face or lead-in: `tip_local()[2]` stops within
~0–2 mm of `seated_depth_m` instead of falling toward 0. The plug is compact, with both lateral extents within a
few millimetres. Validated on a USB-C plug in its receptacle. Pressing and dragging sideways does not search,
because friction pins the pressed tip. Lift, move, press instead.

**Code.** It uses `command()` and `tip_local()` from the template; `math` is imported.

- The approach is 23 ticks, and each probe is ~7 ticks (1 lift, 1 lateral, ~3 descent, 2 still).
- The grid holds ~100 candidates at r = 2.5 mm. With the 0.75 mm thin step, the worst case seen was 429 of 1200
  ticks (733 with the earlier 0.25 mm thin step).
- The plug stays held in the jaws throughout.

```python
md = insertion.get_task_metadata()
DEPTH = float(md["seated_depth_m"])
st = {"face": None, "top": None, "drift": [0.0, 0.0], "pos": [0.0, 0.0, DEPTH + 0.002]}
# Approach: 20-tick ramp to 2 mm over the mouth on the estimate axis, then 3 settle ticks. Required: the first
# probe is the calibration press and must start from rest here, or the measured drift is +-2 mm of noise.
s0 = tip_local()
for i in range(1, 24):
    a = min(i / 20.0, 1.0); command([s0[0] * (1 - a), s0[1] * (1 - a), s0[2] * (1 - a) + (DEPTH + 0.002) * a])

def probe(lat):  # lat = [x, y] in the estimate frame; True once the tip drops into the mouth
    x = [lat[0] - st["drift"][0], lat[1] - st["drift"][1]]  # put the PRESSED tip at lat
    hz = DEPTH + 0.002 if st["top"] is None else st["top"] + 0.0008  # 0.8 mm over the highest stall
    command([st["pos"][0], st["pos"][1], hz]); command([x[0], x[1], hz])
    floor = (DEPTH if st["face"] is None else st["face"]) - 0.0035  # press 3.5 mm past the face
    enter = DEPTH - 0.003 if st["face"] is None else st["face"] - 0.0015  # entry is RELATIVE to the face
    z = hz; last = tip_local(); still = 0
    while still < 2 and int(get_task_context().metadata["episode_steps_remaining"]) > 30:
        z = max(z - 0.0015, floor); command([x[0], x[1], z]); cur = tip_local(); st["pos"] = [x[0], x[1], z]
        if cur[2] < enter:
            return True
        still = still + 1 if abs(cur[2] - last[2]) < 0.00004 and (z <= cur[2] - 0.001 or z <= floor) else 0; last = cur
    if st["face"] is None:  # first stall: face height and the lateral slide of the pressed tip
        st["face"] = cur[2]; st["drift"] = [cur[0] - x[0], cur[1] - x[1]]
    st["top"] = cur[2] if st["top"] is None else max(st["top"], cur[2])
    return False

U = md["uncertainty"]; B = md["plug_bounds_in_tip_frame_m"]
RLIM = 1.25 * (float(U["grasp_lateral_m"]) + float(U["socket_estimate_lateral_m"]))
THIN = 0 if float(B["max"][0]) - float(B["min"][0]) < float(B["max"][1]) - float(B["min"][1]) else 1
STEP = [0.0005, 0.0005]; STEP[THIN] = 0.00075  # 0.5 mm on the wide axis, 0.75 mm on the thin axis
cands = []
for i in range(-int(RLIM / STEP[0]), int(RLIM / STEP[0]) + 1):
    for j in range(-int(RLIM / STEP[1]), int(RLIM / STEP[1]) + 1):
        r = math.hypot(i * STEP[0], j * STEP[1])
        if r <= RLIM:
            cands.append([int(r / 0.0005), math.atan2(j * STEP[1], i * STEP[0]), i * STEP[0], j * STEP[1]])
cands.sort()  # ring by ring outward, by angle inside a ring: consecutive probes are neighbours
hit = None
for c in cands:
    if int(get_task_context().metadata["episode_steps_remaining"]) < 60:
        break
    if probe([c[2], c[3]]):
        hit = c
        break
if hit is not None:  # seat: ramp to 4 mm past the seated estimate at 0.6 mm/tick, hold 20 ticks
    z = st["pos"][2]
    while z > -0.004:
        z = max(z - 0.0006, -0.004); command([st["pos"][0], st["pos"][1], z])
    for k in range(20):
        command([st["pos"][0], st["pos"][1], -0.004])
# hit is None: the grid ran out of candidates or ticks without finding the mouth.
```

**Why it works.**
- Under a press, the tip slides by a fixed, arm-specific amount (~+1 mm wide and −0.2 mm thin on USB-C). Measuring
  that once and subtracting it re-centres the whole search.
- Judging entry relative to the first face contact avoids an absolute threshold that the probe floor can sit above.
- The stall height is flat off the hole, so there is no gradient to follow, and the grid must cover the area.

**Evidence.**

| run | development | unseen | held-out | v00 |
|---|---|---|---|---|
| proposer (usb_c_into_receptacle, v06) | 10/10 | 5/5 | 49/50 | 4/10 |
| verifier, the template plus this text applied literally | 5/5 | – | – | – |
| usb_a_into_receptacle proposer (0.75 mm thin step) | 10/10 | 5/5 | 50/50 | 10/10 |
| edit verifier, USB-A + USB-C seeds 101–105 | 5/5 + 5/5 | – | – | – |

On the verifier's runs, the drift measured +1.05..+1.09 mm wide and −0.11..−0.26 mm thin. The mouth was found after
13–49 probes.

**Transfer.** usb_a_into_receptacle graded this entry `useful`.
- Applied literally, it scored 10/10 on development seeds, where a straight press scored 3/10.
- The drift measured +0.92..+1.12 mm wide and −0.04..−0.29 mm thin, the same as on USB-C.
- With drift calibration off, it scored 9/10.

**Structural transfer.** nvme_into_m2_dock graded this entry `useful` as the structure of its 10/10 program: a
lift-move-press probe, a ring-ordered grid, and judgement relative to the face. It was adapted to an edge card,
with the drift calibration dropped and the entry test replaced by a face-relative seat window.

**Limits.**
- **Calibration matters for the tick budget.** The proposer's ablation with drift calibration off scored 9/10, and
  3 seeds finished with only ~130 ticks to spare. The verifier did not retest this.
- **Grid spacing.** The step was 0.25 mm on the thin axis until usb_a_into_receptacle showed the wide axis was the
  tight one. Coarsening the thin step to 0.75 mm kept 10/10 there and cut the worst case from 725 to 429 ticks.
  An independent verifier then ran the edited grid on USB-A (5/5) and on USB-C (5/5, no regression).
- **Which axis is tight depends on the receptacle, not the plug bounds.** On a new connector, measure it by tripling
  one axis's step at a time on the development seeds.
- **Search radius.** Widening the disc to 1.75× was proposed but is untested: every entry seen lay within 1.6 mm of
  the estimate.
- **The 1.5 mm entry threshold was measured on these receptacles.** Re-measure the drift and the face height on a
  new connector.

## Hard-stop seat test: reject plugs wedged beside the socket

**Trigger.**
- **When to run it.** The tip went past the socket face during a search and stopped near the seat estimate. Run
  this before you accept any searched seat.
- **Why.** A plug wedged under or slid beside a board socket also stops at a plausible depth, and in depth alone it
  looks seated.
- **Inputs.** `lat` is the [x, y] lateral command in the estimate frame where the plug passed the face. It uses
  `command()` and `tip_local()` from the template, so the orientation is Q0.
- **Cost.** 25 ticks of press, plus up to 3 rounds of 10 ticks. The plug stays held.
- Validated on the M.2 board socket (side grasp).

```python
unc = insertion.get_task_metadata()["uncertainty"]
AX_UNC = float(unc["grasp_axial_m"]) + float(unc["socket_estimate_axial_m"])
FLOOR = -(AX_UNC + 0.0005)

def hard_stop_seated(lat):
    for k in range(25):
        command([lat[0], lat[1], FLOOR])
    z1 = tip_local()[2]
    if z1 < FLOOR + 0.0004 or abs(z1) > AX_UNC + 0.0004:
        return False  # followed the command into free space, or stopped too far from the seat
    for rnd in range(3):  # a true seat barely moves under a 2 mm deeper command; a wedged plug yields
        for k in range(10):
            command([lat[0], lat[1], FLOOR - 0.002])
        z2 = tip_local()[2]
        if z1 - z2 < 0.00015:
            return abs(z2) < AX_UNC + 0.0004
        z1 = z2
    return False
```

**Why it works.** A true seat is a hard stop: it moved 0.04–0.10 mm under a 2 mm deeper command. A plug wedged or
slipped beside the socket yields 0.42–6.2 mm. The repeat rounds let a nearly-seated plug settle without being
rejected.

**Evidence.**

| run | development | unseen | held-out | v00 |
|---|---|---|---|---|
| proposer (nvme_into_m2_socket) | 8/10 | 4/5 | eval running | 4/10 |
| verifier | 4/5 (attempt 2) | – | – | – |

- **Proposer.** The test added 2 seeds, and the repeat rounds 1 more.
- **Verifier.** It accepted every true seat it checked (z 0.00–0.19 mm), never a false one, and rejected 20+
  presses beside the socket at 25 ticks each.

**Limits.**
- **Orientation.** The test commands orientation Q0. After a roll or tilt search, run it at the found orientation,
  or it undoes the correction. A roll-aware variant has not been verified.
- **No lateral check.** The test accepted a wedged PCIe card at z 1.27 mm with the tip 6.65 mm off sideways. Add
  `abs(lateral − command) < 2 mm` to its acceptance, as the generalist did.
- **A hard shelf inside the axial window fools it.** The M.2 dock shelf at 0.59 mm passed as a seat. Keep the
  window narrow (0.5 mm) unless the plug is a long card.
- **Not promoted.** Two companion proposals from the same task were graded defective by the verifier and are not
  in the library:
  - **Roll sweep at a partial entry.** The sweep found entries, but nothing confirmed the seat afterwards.
  - **Free-space press probe.** Its face-height prior never reached the face.

## Tall guide posts (DIMM): measure alignment from RGB-D, descend with lift-and-poke, back out and retry

**Trigger.**
- **What you see.** A straight descent stalls well above the mouth, on guide posts that rise above the socket (about
  22 mm above the seat on the DIMM slot).
- **Why pushing harder fails.** It rotates the plug in the grasp and slides it 2–6 mm sideways.
- **Why vision is needed.** The guide slits leave only about ±0.2 mm of x clearance, smaller than the hidden errors,
  so alignment has to be measured before descending. There are no perception services, so back-project RGB-D
  yourself.
- Validated on the DDR4 DIMM slot.

**Code.** This section has three parts:
1. **Back-projection.** `pixel_to_local(cam)` maps (u, v, depth) to the estimate frame. It uses `P0`, `EX`, `EY`,
   `EZ` from "Frame helpers".
   - Read pixels with `cam.depth_m.tolist()` and `cam.rgb.tolist()`.
   - Loop only over the projected bounding box of the region of interest: under 1 s for both 480×360 cameras.
2. **Lift-and-poke descent.** `descend(off)` steps down 0.3 mm/tick. On a stall (the tip trails the command by more
   than 0.8 mm) it lifts and pokes small lateral offsets instead of pushing. Each poke costs 8 ticks.
3. **Back out and re-descend.** When a poke search is exhausted, back out and re-descend from an independent
   alignment estimate: 21 ticks plus 60–250 per attempt.

`command(local, q)` here takes an explicit orientation quaternion.

```python
def qmat(q):
    w, x, y, z = q
    return [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]

def pixel_to_local(cam):
    # camera_pose is in pr_base_link with OpenCV axes (+z forward, +x right, +y down)
    K = cam.intrinsics
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    R = qmat([float(v) for v in cam.camera_pose.quaternion_wxyz])
    t = [float(v) for v in cam.camera_pose.position]
    M = [[sum(E[i] * R[i][k] for i in range(3)) for k in range(3)] for E in (EX, EY, EZ)]
    o = [sum(E[i] * (t[i] - P0[i]) for i in range(3)) for E in (EX, EY, EZ)]
    def f(u, v, d):
        xc, yc = (u - cx) / fx * d, (v - cy) / fy * d
        return [M[k][0] * xc + M[k][1] * yc + M[k][2] * d + o[k] for k in range(3)]
    return f

def qconj(q):
    return [q[0], -q[1], -q[2], -q[3]]

def state_local():
    pose = get_robot_state().end_effector_poses["primary"]
    p = pose.position
    d = [float(p[i]) - P0[i] for i in range(3)]
    loc = [sum(d[i] * axis[i] for i in range(3)) for axis in (EX, EY, EZ)]
    qr = qmul(qconj(Q0), [float(v) for v in pose.quaternion_wxyz])
    s = 1.0 if qr[0] >= 0 else -1.0
    return loc, [2.0 * s * qr[k] for k in (1, 2, 3)]

def ticks_left():
    return get_task_context().metadata["episode_steps_remaining"]

unc = md["uncertainty"]
AX_TOL = float(unc["grasp_axial_m"]) + float(unc["socket_estimate_axial_m"])
SPEED = 0.0003     # 0.3 mm/tick
LAG = 0.0008       # stall: the tip trails the command by more than 0.8 mm
Z0 = state_local()[0][2]   # start height, above the guides
Q = Q0             # replace with the vision-corrected orientation (see Limits)
pattern = [[0, 0.0003], [0, -0.0003], [0.0002, 0], [-0.0002, 0], [0, 0.0006], [0, -0.0006],
           [0.0002, 0.0003], [-0.0002, 0.0003], [0.0002, -0.0003], [-0.0002, -0.0003], [0, 0.001], [0, -0.001],
           [0.0004, 0], [-0.0004, 0], [0, 0.0015], [0, -0.0015]]

def descend(off):
    # Returns (seated, off). Lifts and pokes on each stall instead of pushing harder.
    zc = Z0
    while ticks_left() > 40:
        zc -= SPEED
        command([off[0], off[1], zc], Q)
        t, rv = state_local()
        if t[2] < AX_TOL:
            return True, off
        if t[2] - zc > LAG:
            zs = t[2]
            found = None
            for c in pattern:
                if ticks_left() < 60:
                    break
                x, y = off[0] + c[0], off[1] + c[1]
                for k in range(3):
                    command([x, y, zs + 0.001], Q)
                for k in range(5):
                    command([x, y, zs - 0.0015], Q)
                t2, rv2 = state_local()
                if t2[2] < zs - 0.0007:
                    found = [x, y]
                    break
            if found is None:
                return False, off  # poke pattern exhausted at this gate
            off = found
            zc = state_local()[0][2]
    return False, off

def back_out(off):
    t, rv = state_local()
    for i in range(15):
        a = (i + 1) / 15.0
        command([t[0] + (off[0] - t[0]) * a, t[1] + (off[1] - t[1]) * a, t[2] + (Z0 - t[2]) * a], Q)
    for i in range(6):
        command([off[0], off[1], Z0], Q)

# Usage: seated, off = descend([dx, Y]); if not seated, then for each alternative x estimate xc,
# back_out([xc, Y]) and descend([xc, Y]) again while ticks_left() > 250.
```

**Why it works.**
- On a stall, pushing converts force into in-grasp rotation. Lifting and poking keeps the plug square.
- A dense poke grid at a jammed gate does not help: 63 pokes scored 0 passes. A fresh descent from a different
  alignment estimate does.

**Evidence.**

| run | development | unseen | held-out | v00 |
|---|---|---|---|---|
| proposer (ram_into_dimm_slot) | 10/10 | 3/5 | 49/50 (fixed renderer) | 0/10 |
| verifier | 5/5 (attempt 2) | – | – | – |

- **Back-projection.** The verifier graded all three parts sound. The two cameras agreed to 0.03° on yaw.
- **Lift-and-poke.** 1–59 pokes per seed.
- **Back out and retry.** It decided 3 of the 5 seeds.

**Limits.**
- **These parts need the alignment estimator.** Applied with `Q = Q0` they reproduced only 2/5. Get `dx`, `Y` and
  `Q` from "RGB-D card/slot alignment" below: OFF = (x_cmd, y_cmd), Q = q.
- **Unseen residual.** The proposer's unseen residual was a slit gate ~3 mm below the post tops (2 of 5 seeds).

## Edge card sits on the slot dip floor: wide-axis pecks under load, press on from the catch

**Trigger.**
- **Plug shape.** An edge card: one lateral extent in `plug_bounds_in_tip_frame_m` is 2 mm or less, and the other
  is much larger.
- **Profile.** The thin-axis peck profile (pecks to `seated_depth_m − 3 mm`) reads plateau | dip about 1–1.5 mm
  deep | plateau, with free drops beyond the plateaus (the housing's outer edge, not the slot).
- **Symptom.** Re-pressing in the dip from above never goes deeper. This is the failed precondition of the entry
  above.
- **Validated** on the PCIe riser, and transferred to the PCIe x16 card.

**Code.** It uses `command()` and `tip_local()` from the template.
- Tick cost: 30 approach + 29 × 15 thin scan + up to 21 × 15 wide pecks + 55 press, which is 685–840 of 1200
  ticks.
- The plug stays held throughout.

```python
md = insertion.get_task_metadata()
bmin = md['plug_bounds_in_tip_frame_m']['min']
bmax = md['plug_bounds_in_tip_frame_m']['max']
THIN = 0 if (bmax[0] - bmin[0]) < (bmax[1] - bmin[1]) else 1
LAT = float(md['uncertainty']['grasp_lateral_m']) + float(md['uncertainty']['socket_estimate_lateral_m'])
DEPTH = float(md['seated_depth_m'])
z_hi = DEPTH + 0.0025
PITCH = 0.0003

def lat(t, w, z):
    v = [0.0, 0.0, z]
    v[THIN] = t
    v[1 - THIN] = w
    return v

s0 = tip_local()  # approach: 30-tick ramp to (0, 0, z_hi)
for i in range(30):
    a = (i + 1) / 30.0
    command([s0[0] * (1 - a), s0[1] * (1 - a), s0[2] * (1 - a) + z_hi * a])
n = int(round((2.0 * LAT + 0.0002) / PITCH))  # thin-axis peck scan over twice the lateral uncertainty
prof = []
for i in range(2 * n + 1):
    t = PITCH * (i - n)
    for k in range(5): command(lat(t, 0.0, z_hi))
    for k in range(10): command(lat(t, 0.0, DEPTH - 0.003))
    prof.append([t, tip_local()[2]])
top = max([p[1] for p in prof])
runs = []
cur = []
for i in range(len(prof)):
    if prof[i][1] < top - 0.0004:
        cur.append(i)
    else:
        if cur:
            runs.append(cur)
        cur = []
inner = [r for r in runs if r[0] > 0]  # a run starting at the scan edge is the free drop, not the slot
target = None
floor = None
entered = None
if inner:
    floor = min([prof[i][1] for i in inner[0]])
    deep = [prof[i][0] for i in inner[0] if prof[i][1] < floor + 0.0002]
    target = 0.5 * (min(deep) + max(deep))  # aim at the deepest samples, not the run centre
    nx = int(round((LAT + 0.0015) / 0.00035))
    for j in range(2 * nx + 1):  # wide-axis pecks UNDER LOAD: 1.5 mm past the dip floor
        w = 0.00035 * (j - nx)
        for k in range(5): command(lat(target, w, z_hi))
        for k in range(10): command(lat(target, w, floor - 0.0015))
        if tip_local()[2] < floor - 0.0008:
            entered = w
            break
if entered is not None:  # press on from the catch WITHOUT lifting, to 3 mm past the seat estimate
    z0 = floor - 0.0015
    for k in range(40): command(lat(target, entered, z0 + (-0.003 - z0) * (k + 1) / 40.0))
    for k in range(15): command(lat(target, entered, -0.003))
# inner empty (no slot dip found) or entered is None: no fallback yet (see Limits).
```

**Why it works.**
- The plug only "catches" the slot key while it is loaded against the dip floor. Under that load the measured tip
  lags the commanded wide offset by ~1.4 mm.
- Lifting and re-pressing at the catch offset re-entered 0/9 times.
- The slot floor is the true seat, so the final press goes past the estimate.

**Evidence.**

| run | development | unseen | held-out | v00 |
|---|---|---|---|---|
| proposer (pcie_riser_into_x16_slot) | 10/10 | 4/5 | eval running | 0/10 (with the edge-card entry above) |
| verifier, this snippet as written | 5/5 | – | – | – |
| transfer: verifier's same program on pcie_card_into_x16_slot | 4/5 | – | – | – |

Proposer ablation: pressing only 0.8 mm past the floor during the wide pecks scored 7/10.

**Limits.**
- **No fallback when no inner dip is found.** In the PCIe card transfer, one seed's only low run started at the scan
  edge, so the program never pressed.
- **Narrow dips.** A dip about 0.6 mm wide (2 samples) lost the proposer one unseen seed: the loaded pecks dragged
  the tip onto the wall edge.
- **As a fallback on the PCIe x16 card it added nothing.** The card's phase-3 rerun graded this entry `misleading`:
  chained after the housing-top re-presses, it seated 0 of the 3 seeds it ran on.
  - On a card that was already wedged in the slot, the first loaded peck read as a false catch.
  - The verifier's standalone transfer to the card (4/5) and this fallback result disagree. That is unresolved.
- **The press-on has no seat check.** A false catch at the edge of the sweep was pressed on and ended 4.1 mm off
  sideways (unseen PCIe). Gate the finish with the lateral check.

## Edge card in a recessed connector: gentle 2-D peck grid, seat = face − seated_depth

**Trigger.**
- **Profile.** A straight press ("First try") stalls near the mouth, and the thin-axis peck scan finds no dip inside
  the plateau: every press lands on one flat face height, and drops appear only at its edges. This is the failed
  precondition of "Edge card stalls on the housing top".
- **Why depth alone misleads.** Those edge drops are features beside the connector: a shelf at face − 1.75 mm and a
  cavity at face − 5..7 mm on the M.2 dock. So classify each stall relative to the measured face, never by
  absolute depth.
- **Validated** on the M.2 card into a recessed dock connector.

**Code.** It uses `command()` and `tip_local()` from the template, plus `math`.
- Tick cost: 26 approach + ~9 per probe + 20 to finish. The full 2.5 mm grid at 0.5 mm is ~79 probes (~740
  ticks); the worst case seen was 562 of 1200.
- The plug stays held throughout.

```python
md = insertion.get_task_metadata()
DEPTH = float(md["seated_depth_m"])
U = md["uncertainty"]
AX_UNC = float(U["grasp_axial_m"]) + float(U["socket_estimate_axial_m"])
RLIM = 1.25 * (float(U["grasp_lateral_m"]) + float(U["socket_estimate_lateral_m"]))
st = {"face": None, "pos": [0.0, 0.0, DEPTH + 0.002], "stalls": []}

def is_seat(z):  # the seat is seated_depth below the connector face, not a fixed height
    return st["face"] is not None and abs(z - (st["face"] - DEPTH)) < 0.00035

def probe(lat):  # lift out, move in free space, gentle press (cmd <= 1.5 mm below tip, 1 mm/tick)
    hz = DEPTH + 0.002 if st["face"] is None else st["face"] + 0.0008
    for k in range(5):
        command([st["pos"][0], st["pos"][1], hz])
        if tip_local()[2] > hz - 0.0005:
            break
    command([lat[0], lat[1], hz])
    fl = -(AX_UNC + 0.0015) if st["face"] is None else st["face"] - DEPTH - 0.001
    z = hz; cur = tip_local(); still = 0; n = 0
    while still < 2 and n < 12 and cur[2] > fl + 0.0003:
        prev = cur; z = max(z - 0.001, cur[2] - 0.0015, fl)
        command([lat[0], lat[1], z]); cur = tip_local(); n += 1
        still = still + 1 if abs(cur[2] - prev[2]) < 0.00004 else 0
    st["pos"] = [lat[0], lat[1], z]
    if abs(cur[2] - DEPTH) < 0.0013 and (st["face"] is None or cur[2] > st["face"]):
        st["face"] = cur[2]  # face = highest stall within 1.3 mm of the mouth estimate
    st["stalls"].append([lat[0], lat[1], cur[2]])
    return is_seat(cur[2])

s0 = tip_local()  # approach: 20-tick ramp onto the estimate axis, 2 mm above the mouth, + 3 settle ticks
for i in range(1, 24):
    a = min(i / 20.0, 1.0); command([s0[0] * (1 - a), s0[1] * (1 - a), s0[2] * (1 - a) + (DEPTH + 0.002) * a])
S = 0.0005
N = int(RLIM / S)
cands = sorted([[round(math.hypot(i, j) * S / RLIM * 8), math.atan2(j, i), i * S, j * S]
                for i in range(-N, N + 1) for j in range(-N, N + 1) if math.hypot(i, j) * S <= RLIM])
hit = None
for c in cands:
    if int(get_task_context().metadata["episode_steps_remaining"]) < 45:
        break
    if probe([c[2], c[3]]):
        hit = [c[2], c[3]]; break
    old = [s for s in st["stalls"][:-1] if is_seat(s[2])]  # seat pressed before the face was known
    if old:
        if probe([old[0][0], old[0][1]]):
            hit = [old[0][0], old[0][1]]; break
        st["stalls"] = [s for s in st["stalls"] if not is_seat(s[2])]
if hit is not None:  # finish: 20 ticks of gentle press at the hit
    zs = st["pos"][2]
    for k in range(20):
        # As verified, this press has no depth limit (see Limits). It follows the tip down wherever it can slip.
        zs = max(tip_local()[2] - 0.001, zs - 0.0005); command([hit[0], hit[1], zs])
```

**Why it works.**
- The face is the highest stall within 1.3 mm of the mouth estimate.
- The seat is the one stall within 0.35 mm of face − `seated_depth_m`, so it doesn't matter where the face actually
  sits: on two of the verifier's seeds it was ~0.9 mm below the estimate.
- The gentle press (command at most 1.5 mm below the tip, 1 mm/tick) avoids the 2–3 mm sideways deflection that
  deep over-presses caused.

**Evidence.**

| run | development | unseen | held-out | v00 |
|---|---|---|---|---|
| proposer (nvme_into_m2_dock) | 10/10 | 5/5 | 50/50 | 3/10 |
| verifier, snippet appended word for word | 5/5 (worst 562 ticks) | – | – | – |
| transfer: same program on nvme_into_m2_socket | 4/5 | – | – | – |

On the verifier's runs, seat − face measured −3.17..−3.21 mm against `seated_depth_m` = 3.2 mm. The proposer's
ablation without the face-relative seat window scored 9/10.

**Limits.**
- **The final press has no depth limit.** In the M.2 socket transfer it followed the tip down to −4.87 mm after a
  seat was accepted at −0.81 mm. Capping the finish at about face − `seated_depth_m` − 1 mm, or gating it with
  "Hard-stop seat test", is the suggested fix. Neither has been verified.
- **The search radius is 1.25× the summed lateral uncertainty** (2.5 mm here). A slot beyond that exhausts the grid.
- **Misleading on the M.2 board socket.** There the face is not `seated_depth_m` above the seat, so the
  face-relative rule missed true seats and accepted false ones. Gate every seat with "Hard-stop seat test", and
  reject a hit whose capped finish sinks to the cap; then keep searching. Even so, the generalist reached only
  8/10 dev and 37/50 held-out there.
- **Don't carry the gentle press over to edge-card re-presses.** On the PCIe x16 card, capping the re-press at
  1.5 mm below the tip scored 6/10 against 8/10 with the forceful press to DEPTH − 8 mm. The card needs that force
  to pass the slot key. This came from the card's phase-3 rerun, which graded this entry `misleading` for that
  transfer.

## RGB-D card/slot alignment: yaw, lean and offset for long edge cards and tall guides

**Trigger.** Run this for a long edge card, where the yaw error over half its length exceeds its thickness (see
"Routing"), or after a stall high on tall guide posts. Pass the measured tip, held at rest at orientation Q0 for
10+ ticks.

It returns `[q, x_cmd, y_cmd, xs]`, or `None` if its quality gates fail. In that case it switches off silently;
keep Q0.
- `q`: the corrected command orientation.
- `x_cmd`: the wide-axis command; 0 when there are no guide posts.
- `y_cmd`: the thin-axis command.
- `xs`: the per-camera x values, used for retries.

**Requires.**
- The template's `P0`, `EX`, `EY`, `EZ`, `Q0` and `qmul`, plus `math`.
- `DEPTH = seated_depth_m`.
- `BMIN`/`BMAX` = `plug_bounds_in_tip_frame_m` min/max.
- `EXT = [BMAX[0]-BMIN[0], BMAX[1]-BMIN[1]]` and `THIN = 0 if EXT[0] < EXT[1] else 1`.

**Cost.** It costs zero ticks: about 1 s of compute for two 480×360 cameras.

**Colour classes.** They are scene knowledge, gated by fit quality:
- the card is green PCB;
- the socket housing top is dark;
- the guide posts are white.

```python
def qmat(q):
    w, x, y, z = q
    return [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))]


def solve3(A, b):
    # Gaussian elimination for a 3x3 system
    M = [list(A[i]) + [b[i]] for i in range(3)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(4)]
    return [M[i][3] / M[i][i] for i in range(3)]


def line_fit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((xs[i] - mx) ** 2 for i in range(n))
    if sxx <= 0:
        return None
    b = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
    return [my - b * mx, b]


def guide_vision(tip):
    # Back-project both cameras into the estimate frame (absolute, metres); measure the card (green PCB
    # points inside the plug's bounds box), the housing top at mouth height (dark) and guide posts (white).
    obs = get_observation()
    per = []
    xlo = tip[0] + BMIN[0] - 0.003
    xhi = tip[0] + BMAX[0] + 0.003
    for name in obs.cameras:
        cam = obs.cameras[name]
        K = cam.intrinsics
        fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
        Rc = qmat([float(v) for v in cam.camera_pose.quaternion_wxyz])
        tc = [float(v) for v in cam.camera_pose.position]
        M = [[sum(E[i] * Rc[i][k] for i in range(3)) for k in range(3)] for E in (EX, EY, EZ)]
        o = [sum(E[i] * (tc[i] - P0[i]) for i in range(3)) for E in (EX, EY, EZ)]
        dep = cam.depth_m.tolist()
        rgb = cam.rgb.tolist()
        green = []
        dark = []
        wl = []
        wr = []
        for v in range(len(dep)):
            row = dep[v]
            for u in range(len(row)):
                d = row[u]
                if d <= 0 or d > 3:
                    continue
                xc, yc = (u - cx) / fx * d, (v - cy) / fy * d
                px = M[0][0] * xc + M[0][1] * yc + M[0][2] * d + o[0]
                if px < xlo - 0.01 or px > xhi + 0.01:
                    continue
                py = M[1][0] * xc + M[1][1] * yc + M[1][2] * d + o[1]
                if abs(py) > 0.02:
                    continue
                pz = M[2][0] * xc + M[2][1] * yc + M[2][2] * d + o[2]
                if pz < -0.01 or pz > tip[2] + 0.05:
                    continue
                c = rgb[v][u]
                r, g, b = int(c[0]), int(c[1]), int(c[2])
                if g > r + 15 and g > b + 10:
                    if xlo <= px <= xhi and pz > tip[2] - 0.002 and abs(py - tip[1]) < 0.004:
                        green.append([px, py, pz])
                elif r < 45 and g < 45 and b < 45 and DEPTH - 0.0015 < pz < DEPTH + 0.002:
                    dark.append([px, py])
                elif r > 150 and g > 150 and b > 150 and pz < tip[2]:
                    if px < tip[0] + 0.5 * (BMIN[0] + BMAX[0]):
                        wl.append(px)
                    else:
                        wr.append(px)
        if len(green) < 200 or len(dark) < 200:
            continue
        A = [[0.0] * 3 for i in range(3)]
        bb = [0.0] * 3
        for p in green:
            f = [1.0, p[0], p[2] - tip[2]]
            for i in range(3):
                bb[i] += f[i] * p[1]
                for j in range(3):
                    A[i][j] += f[i] * f[j]
        card = solve3(A, bb)
        if card is None:
            continue
        rms = math.sqrt(sum((p[1] - card[0] - card[1] * p[0] - card[2] * (p[2] - tip[2])) ** 2 for p in green) / len(green))
        gx = [p[0] for p in green]
        ends = [pct(gx, 0.5), pct(gx, 99.5)]
        bx, blo, bhi = [], [], []
        for k in range(int((xhi - xlo) / 0.01)):
            lo = xlo + 0.01 * k
            ys = [p[1] for p in dark if lo <= p[0] < lo + 0.01]
            if len(ys) > 30:
                bx.append(lo + 0.005)
                blo.append(pct(ys, 3))
                bhi.append(pct(ys, 97))
        if len(bx) < 3:
            continue
        # Keep full bins (the housing ends and occluded bins are partial: fewer points, odd width).
        widths = [bhi[i] - blo[i] for i in range(len(bx))]
        wmed = sorted(widths)[len(widths) // 2]
        cmax = max([len([p for p in dark if bx[i] - 0.005 <= p[0] < bx[i] + 0.005]) for i in range(len(bx))])
        keep = [i for i in range(len(bx)) if abs(widths[i] - wmed) < 0.0006
                and len([p for p in dark if bx[i] - 0.005 <= p[0] < bx[i] + 0.005]) >= 0.4 * cmax]
        if len(keep) < 3:
            continue
        edges = []
        for ys in (blo, bhi):
            kx = [bx[i] for i in keep]
            ky = [ys[i] for i in keep]
            fl = line_fit(kx, ky)
            res = [abs(ky[i] - fl[0] - fl[1] * kx[i]) for i in range(len(kx))]
            worst = max(range(len(kx)), key=lambda i: res[i])
            if len(kx) > 3 and res[worst] > 0.0003:
                fl = line_fit([kx[i] for i in range(len(kx)) if i != worst], [ky[i] for i in range(len(kx)) if i != worst])
            edges.append(fl)
        width = wmed
        posts = None
        if len(wl) > 20 and len(wr) > 20:
            posts = 0.5 * (pct(wl, 97) + pct(wr, 3))
        per.append({"a": card[0], "yaw": card[1], "lean": card[2], "rms": rms, "ends": ends, "width": width,
                    "slot_y": 0.5 * (edges[0][0] + edges[1][0]), "edge_yaws": [edges[0][1], edges[1][1]],
                    "posts": posts, "n": [len(green), len(dark), len(keep)]})
    return per


def vision_alignment(tip):
    # Returns [Q, x_cmd, y_cmd, x_alternatives, info] or None if the measurement is not trustworthy.
    per = [p for p in guide_vision(tip) if p["rms"] < 0.0008 and 2.0 * EXT[THIN] < p["width"] < 12.0 * EXT[THIN]]
    if not per:
        return None
    if len(per) > 1 and abs(per[0]["yaw"] - per[1]["yaw"]) > 0.004:
        return None
    n = float(len(per))
    a = sum(p["a"] for p in per) / n - tip[1]
    yaw = sum(p["yaw"] for p in per) / n
    lean = sum(p["lean"] for p in per) / n
    sy = sum(p["slot_y"] for p in per) / n
    ey = sorted([e for p in per for e in p["edge_yaws"]])
    syaw = 0.5 * (ey[(len(ey) - 1) // 2] + ey[len(ey) // 2])  # median edge slope over both edges and cameras
    th = syaw - yaw
    ph = math.atan(lean)
    q = qmul(qmul(Q0, [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]), [math.cos(ph / 2), math.sin(ph / 2), 0.0, 0.0])
    ycmd = sy - a
    xs = [p["posts"] - (0.5 * (p["ends"][0] + p["ends"][1]) - tip[0]) for p in per if p["posts"] is not None]
    xcmd = sum(xs) / len(xs) if xs else 0.0
    info = {"cams": len(per), "card_mm": round(a * 1e3, 2), "yaw_mrad": round(yaw * 1e3, 1), "lean_mrad": round(lean * 1e3, 1),
            "slot_y_mm": round(sy * 1e3, 2), "slot_yaw_mrad": round(syaw * 1e3, 1),
            "x_mm": [round(v * 1e3, 2) for v in xs], "cmd_mm": [round(xcmd * 1e3, 2), round(ycmd * 1e3, 2)]}
    return [q, xcmd, ycmd, xs, info]
```

**How to use it.**
- **Tall guides (DIMM).** After the high stall, back out to the start height at Q0, call
  `vision_alignment(tip_local())`, then run "Tall guide posts" with `descend([x_cmd, y_cmd])` and `Q = q`. Retry
  with each single camera's x, then 0, then x_cmd ± 0.4 mm.
- **Long cards (PCIe, riser).** Measure once after the lateral alignment at the start height. Then issue every
  later command at the corrected offset and `q`, so the edge-card searches run unchanged around the corrected
  centre.

**Evidence.**

| run | result |
|---|---|
| DIMM verifier (attempt 2, its own implementation of this idea) | 5/5 |
| generalist (from the library's prose alone): DIMM, dev | 3/10 → 10/10 |
| generalist: DIMM, unseen | 5/5 |
| generalist: PCIe card, dev | 7/10 → 9/10 (10/10 with the long-card partial-seat change) |
| generalist's whole-library program: PCIe card, held-out | 50/50 |

- Two agents independently reproduced the method from text.
- This exact code is the generalist's. Its quality gates passed on all 10 riser dev seeds, and the riser stayed
  10/10.
- Measured values: card yaw relative to the command −15..+15 mrad; slot yaw −9..+16 mrad; mismatch on the PCIe
  failures 15–30 mrad; lean −12..+15 mrad. The two cameras agree on card yaw to within 0.5 mrad.

**Limits.**
- **The colour thresholds are specific to these scenes.**
- **The card box filter is essential.** A board or parts of the riser are also green.
- **Only full bins.** Without the full-bin filter, the slot yaw was off by up to 25 mrad.
