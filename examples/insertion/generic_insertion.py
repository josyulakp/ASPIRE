# Generic no-force insertion program for all 8 insertion tasks (ASPIRE campaign 2026-10-20, phase-4 generalist).
# Written from the insertion skill library alone; held-out seeds 1-50: 373/400. Routes on plug bounds and measurements only.
import math
# Generalist v04 (from v03): v02 + short edge cards (yaw lever below the card thickness) skip the thin-axis scan and
# go straight to the recessed 2-D grid (face seeded from the first press); in the grid, a stall past the
# face inside the axial window is verified with the hard-stop test (unless it sits >0.8 mm above face-depth);
# a face-relative hit whose capped finish press sinks to the cap is rejected and the grid continues.
# v04: a long edge card whose first press stops inside the axial window (0.5..1.4 mm) is hard-stop tested
# there (the slot floor is the true seat for long cards; no shelf near the mouth).
# Route: approach + straight press (First try) -> seated? -> high stall (tall guides) ->
# compact plug (peck probe grid) | edge card (thin scan -> dip: loaded wide pecks / no dip: 2-D face-relative grid).

md = insertion.get_task_metadata()
seat = md["seated_tip_pose_estimate"]
FRAME = seat["frame"]
P0 = list(seat["position"])
Q0 = list(seat["quaternion_wxyz"])
U = md["uncertainty"]
DEPTH = float(md["seated_depth_m"])
AX_UNC = float(U["grasp_axial_m"]) + float(U["socket_estimate_axial_m"])
LAT_UNC = float(U["grasp_lateral_m"]) + float(U["socket_estimate_lateral_m"])
BMIN = [float(v) for v in md["plug_bounds_in_tip_frame_m"]["min"]]
BMAX = [float(v) for v in md["plug_bounds_in_tip_frame_m"]["max"]]
EXT = [BMAX[0] - BMIN[0], BMAX[1] - BMIN[1]]
THIN = 0 if EXT[0] < EXT[1] else 1
EDGE_CARD = EXT[THIN] <= 0.0021 and EXT[1 - THIN] >= 4.0 * EXT[THIN]
# Long edge card: a yaw error swings its ends by more than its own thickness.
LONG_CARD = EDGE_CARD and 0.5 * EXT[1 - THIN] * (float(U["grasp_yaw_rad"]) + float(U["socket_estimate_yaw_rad"])) > EXT[THIN]

R = {"route": [], "edge_card": EDGE_CARD, "thin_axis": THIN, "failure_stage": None}
result = R


def qmul(a, b):
    return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]]


def qrot(q, v):
    p = qmul(qmul(q, [0.0, v[0], v[1], v[2]]), [q[0], -q[1], -q[2], -q[3]])
    return [p[1], p[2], p[3]]


EX, EY, EZ = qrot(Q0, [1, 0, 0]), qrot(Q0, [0, 1, 0]), qrot(Q0, [0, 0, 1])


def to_base(local):
    return [P0[i] + EX[i]*local[0] + EY[i]*local[1] + EZ[i]*local[2] for i in range(3)]


# Working frame: estimate frame shifted laterally by OFF and commanded at orientation QC
# (both stay at zero / Q0 unless the RGB-D alignment measured a correction).
OFF = [0.0, 0.0]
QC = [list(Q0)]


def tip_abs():
    p = get_robot_state().end_effector_poses["primary"].position
    d = [float(p[i]) - P0[i] for i in range(3)]
    return [sum(d[i]*axis[i] for i in range(3)) for axis in (EX, EY, EZ)]


def tip_local():
    t = tip_abs()
    return [t[0] - OFF[0], t[1] - OFF[1], t[2]]


def ticks_left():
    return int(get_task_context().metadata["episode_steps_remaining"])


def command(local, q=None):
    if ticks_left() <= 1:
        return False
    tgt = to_base([local[0] + OFF[0], local[1] + OFF[1], local[2]])
    ik = solve_ik(Pose(position=tgt, quaternion_wxyz=(QC[0] if q is None else q), frame=FRAME))
    if not ik.ok:
        return False
    qj = [float(v) for v in ik.joint_positions]
    return step(RobotAction({"primary": ArmCommand("joint_position", qj, embodiment="insertion")})).ok


def lat(t, w, z):
    v = [0.0, 0.0, z]
    v[THIN] = t
    v[1 - THIN] = w
    return v


def seated_here(cmd_lat):
    m = tip_local()
    return m[2] < 0.0005 and abs(m[0] - cmd_lat[0]) < 0.002 and abs(m[1] - cmd_lat[1]) < 0.002


# ---------------------------------------------------------------- RGB-D alignment helpers
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


# ---------------------------------------------------------------- approach + first try
START = tip_local()
R["start_local_mm"] = [round(v * 1e3, 2) for v in START]
z_hi = DEPTH + 0.002
z_top = max(START[2], z_hi)
for i in range(20):  # align laterally at (or above) the start height
    a = (i + 1) / 20.0
    command([START[0] * (1 - a), START[1] * (1 - a), START[2] + (z_top - START[2]) * a])
for k in range(5):
    command([0.0, 0.0, z_top])
# Long edge card: measure and correct yaw/lean/offset from RGB-D before pressing
# (the tall-guide estimator, applied up front).
if LONG_CARD:
    try:
        va = vision_alignment(tip_abs())
    except Exception as exc:
        va = None
        R["vision_error"] = str(exc)[:200]
    if va is not None:
        QC[0] = va[0]
        OFF[0] = va[1]
        OFF[1] = va[2]
        R["start_vision"] = va[4]
        s1 = tip_local()
        for i in range(15):
            a = (i + 1) / 15.0
            command([s1[0] * (1 - a), s1[1] * (1 - a), z_top])
        for k in range(5):
            command([0.0, 0.0, z_top])
zc = z_top
high_stall = None
while zc > z_hi:  # slow descent with lag-based stall detection (tall guides)
    zc = max(z_hi, zc - 0.0003)
    command([0.0, 0.0, zc])
    t = tip_local()
    if t[2] - zc > 0.0008 and t[2] > z_hi + 0.0005:
        high_stall = t[2]
        break
R["high_stall_mm"] = None if high_stall is None else round(high_stall * 1e3, 2)

Z_END = -AX_UNC
first_z = None
if high_stall is None:
    for k in range(5):
        command([0.0, 0.0, z_hi])
    z = z_hi
    while z > Z_END:
        z = max(Z_END, z - 0.0005)
        command([0.0, 0.0, z])
    for k in range(10):
        command([0.0, 0.0, Z_END])
    fm = tip_local()
    first_z = fm[2]
    R["first_press_z_mm"] = round(first_z * 1e3, 2)
    R["first_press_mm"] = [round(v * 1e3, 2) for v in fm]


# ---------------------------------------------------------------- hard-stop seat test
FLOOR_HS = -(AX_UNC + 0.0005)


def hard_stop_seated(latp):
    for k in range(25):
        command([latp[0], latp[1], FLOOR_HS])
    m = tip_local()
    z1 = m[2]
    log = [round(v * 1e3, 2) for v in m]
    R.setdefault("hs", []).append(log)
    if z1 < FLOOR_HS + 0.0004 or abs(z1) > AX_UNC + 0.0004:
        return False
    for rnd in range(3):
        for k in range(10):
            command([latp[0], latp[1], FLOOR_HS - 0.002])
        z2 = tip_local()[2]
        log.append(round(z2 * 1e3, 2))
        if z1 - z2 < 0.00015:
            m = tip_local()
            return abs(z2) < AX_UNC + 0.0004 and abs(m[0] - latp[0]) < 0.002 and abs(m[1] - latp[1]) < 0.002
        z1 = z2
    return False


# ---------------------------------------------------------------- compact: peck probe ring grid
def compact_search():
    st = {"face": None, "top": None, "drift": [0.0, 0.0], "pos": [0.0, 0.0, DEPTH + 0.002]}
    s0 = tip_local()
    for i in range(1, 24):
        a = min(i / 20.0, 1.0)
        command([s0[0] * (1 - a), s0[1] * (1 - a), s0[2] * (1 - a) + (DEPTH + 0.002) * a])

    def probe(lp):
        x = [lp[0] - st["drift"][0], lp[1] - st["drift"][1]]
        hz = DEPTH + 0.002 if st["top"] is None else st["top"] + 0.0008
        command([st["pos"][0], st["pos"][1], hz])
        command([x[0], x[1], hz])
        floor = (DEPTH if st["face"] is None else st["face"]) - 0.0035
        enter = DEPTH - 0.003 if st["face"] is None else st["face"] - 0.0015
        z = hz
        last = tip_local()
        cur = last
        still = 0
        while still < 2 and ticks_left() > 30:
            z = max(z - 0.0015, floor)
            command([x[0], x[1], z])
            cur = tip_local()
            st["pos"] = [x[0], x[1], z]
            if cur[2] < enter:
                return True
            still = still + 1 if abs(cur[2] - last[2]) < 0.00004 and (z <= cur[2] - 0.001 or z <= floor) else 0
            last = cur
        if st["face"] is None:
            st["face"] = cur[2]
            st["drift"] = [cur[0] - x[0], cur[1] - x[1]]
        st["top"] = cur[2] if st["top"] is None else max(st["top"], cur[2])
        return False

    RLIM = 1.25 * LAT_UNC
    STEP = [0.0005, 0.0005]
    STEP[THIN] = 0.00075
    cands = []
    for i in range(-int(RLIM / STEP[0]), int(RLIM / STEP[0]) + 1):
        for j in range(-int(RLIM / STEP[1]), int(RLIM / STEP[1]) + 1):
            r = math.hypot(i * STEP[0], j * STEP[1])
            if r <= RLIM:
                cands.append([int(r / 0.0005), math.atan2(j * STEP[1], i * STEP[0]), i * STEP[0], j * STEP[1]])
    cands.sort()
    hit = None
    n = 0
    for c in cands:
        if ticks_left() < 60:
            break
        n += 1
        if probe([c[2], c[3]]):
            hit = c
            break
    R["probes"] = n
    R["drift_mm"] = [round(v * 1e3, 2) for v in st["drift"]]
    R["face_mm"] = None if st["face"] is None else round(st["face"] * 1e3, 2)
    if hit is None:
        R["failure_stage"] = "compact_grid_exhausted"
        return False
    z = st["pos"][2]
    while z > -0.004:
        z = max(z - 0.0006, -0.004)
        command([st["pos"][0], st["pos"][1], z])
    for k in range(20):
        command([st["pos"][0], st["pos"][1], -0.004])
    R["hit_mm"] = [round(hit[2] * 1e3, 2), round(hit[3] * 1e3, 2)]
    return True


# ---------------------------------------------------------------- edge card: recessed 2-D grid
def recessed_grid(face0):
    RLIM = 1.25 * LAT_UNC
    st = {"face": face0, "pos": [0.0, 0.0, DEPTH + 0.002], "stalls": [], "tests": 0, "hs_hit": False}

    def is_seat(z):
        return st["face"] is not None and abs(z - (st["face"] - DEPTH)) < 0.00035

    def probe(lp):
        hz = DEPTH + 0.002 if st["face"] is None else st["face"] + 0.0008
        for k in range(5):
            command([st["pos"][0], st["pos"][1], hz])
            if tip_local()[2] > hz - 0.0005:
                break
        command([lp[0], lp[1], hz])
        fl = -(AX_UNC + 0.0015) if st["face"] is None else st["face"] - DEPTH - 0.001
        z = hz
        cur = tip_local()
        still = 0
        n = 0
        while still < 2 and n < 12 and cur[2] > fl + 0.0003:
            prev = cur
            z = max(z - 0.001, cur[2] - 0.0015, fl)
            command([lp[0], lp[1], z])
            cur = tip_local()
            n += 1
            still = still + 1 if abs(cur[2] - prev[2]) < 0.00004 else 0
        st["pos"] = [lp[0], lp[1], z]
        if abs(cur[2] - DEPTH) < 0.0013 and (st["face"] is None or cur[2] > st["face"]):
            st["face"] = cur[2]
        st["stalls"].append([lp[0], lp[1], cur[2]])
        if is_seat(cur[2]):
            return True
        # Past the face and inside the axial window, but off the face-relative seat: a board socket's face
        # need not sit seated_depth above its seat, so verify the stop with the hard-stop test.
        if (st["face"] is not None and cur[2] > fl + 0.0003 and cur[2] < st["face"] - 0.001
                and abs(cur[2]) < AX_UNC + 0.0004 and cur[2] < st["face"] - DEPTH + 0.0008
                and st["tests"] < 6 and ticks_left() > 150):
            st["tests"] += 1
            if hard_stop_seated(lp):
                st["hs_hit"] = True
                return True
        return False

    s0 = tip_local()
    for i in range(1, 24):
        a = min(i / 20.0, 1.0)
        command([s0[0] * (1 - a), s0[1] * (1 - a), s0[2] * (1 - a) + (DEPTH + 0.002) * a])
    S = 0.0005
    N = int(RLIM / S)
    cands = sorted([[round(math.hypot(i, j) * S / RLIM * 8), math.atan2(j, i), i * S, j * S]
                    for i in range(-N, N + 1) for j in range(-N, N + 1) if math.hypot(i, j) * S <= RLIM])
    hit = None
    n = 0
    rejected = [0]

    def finish(h):
        # Gentle finish press, capped 1 mm below the face-relative seat; a plug that follows the press
        # down to the cap was not on a hard seat (slid beside the socket), so the hit is rejected.
        if st["hs_hit"]:
            return True
        zs = st["pos"][2]
        zcap = st["face"] - DEPTH - 0.001
        for k in range(20):
            zs = max(tip_local()[2] - 0.001, zs - 0.0005, zcap)
            command([h[0], h[1], zs])
        if tip_local()[2] < st["face"] - DEPTH - 0.0006:
            rejected[0] += 1
            return False
        return True

    for c in cands:
        if ticks_left() < 45:
            break
        n += 1
        if probe([c[2], c[3]]):
            if finish([c[2], c[3]]):
                hit = [c[2], c[3]]
                break
            continue
        old = [s for s in st["stalls"][:-1] if is_seat(s[2])]
        if old:
            if probe([old[0][0], old[0][1]]) and finish([old[0][0], old[0][1]]):
                hit = [old[0][0], old[0][1]]
                break
            st["stalls"] = [s for s in st["stalls"] if not is_seat(s[2])]
    R["grid_probes"] = n
    R["grid_tests"] = st["tests"]
    R["grid_rejected"] = rejected[0]
    R["grid_face_mm"] = None if st["face"] is None else round(st["face"] * 1e3, 2)
    if hit is None:
        R["failure_stage"] = "recessed_grid_exhausted"
        return False
    R["hit_mm"] = [round(hit[0] * 1e3, 2), round(hit[1] * 1e3, 2)]
    if st["hs_hit"]:
        R["route"].append("grid_hard_stop")
    return True


# ---------------------------------------------------------------- edge card: thin scan + loaded wide pecks
def edge_card_search():
    s0 = tip_local()
    zh = DEPTH + 0.0025
    for i in range(30):
        a = (i + 1) / 30.0
        command([s0[0] * (1 - a), s0[1] * (1 - a), s0[2] * (1 - a) + zh * a])
    PITCH = 0.0003
    ZSCAN = DEPTH - 0.003
    n = int(round((2.0 * LAT_UNC + 0.0002) / PITCH))
    prof = []
    for i in range(2 * n + 1):
        t = PITCH * (i - n)
        for k in range(5):
            command(lat(t, 0.0, zh))
        for k in range(10):
            command(lat(t, 0.0, ZSCAN))
        prof.append([t, tip_local()[2]])
    R["scan_mm"] = [round(p[1] * 1e3, 2) for p in prof]

    # 1. Scan points that stopped near the seat estimate (not at the command floor): verify with the hard-stop test.
    near = [i for i in range(len(prof)) if abs(prof[i][1]) < AX_UNC + 0.0004 and abs(prof[i][1] - ZSCAN) > 0.00015]
    groups = []
    for i in near:
        if groups and groups[-1][-1] == i - 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    best = sorted([min([[abs(prof[i][1]), i] for i in g]) for g in groups])
    R["near_seat_tests"] = 0
    for b in best[:3]:
        if ticks_left() < 200:
            break
        t = prof[b[1]][0]
        R["near_seat_tests"] += 1
        for k in range(5):
            command(lat(t, 0.0, zh))
        for k in range(10):
            command(lat(t, 0.0, ZSCAN))
        lp = lat(t, 0.0, 0.0)
        if hard_stop_seated([lp[0], lp[1]]):
            R["route"].append("near_seat")
            return True
        for k in range(5):
            command(lat(t, 0.0, zh))

    # 2. Slot dip inside the housing plateau -> loaded wide pecks (0.4 mm dip, then a 0.1 mm dip).
    top = max([p[1] for p in prof])
    inner = []
    for thr in (0.0004, 0.0001):
        runs = []
        cur = []
        for i in range(len(prof)):
            if prof[i][1] < top - thr:
                cur.append(i)
            else:
                if cur:
                    runs.append(cur)
                cur = []
        inner = [r for r in runs if r[0] > 0 and r[-1] < len(prof) - 1]
        if inner:
            R["dip_thr_mm"] = thr * 1e3
            break
    R["inner_runs"] = len(inner)
    if not inner:
        R["route"].append("recessed_grid")
        face0 = top if abs(top - DEPTH) < 0.0013 else None
        return recessed_grid(face0)
    R["route"].append("loaded_wide_pecks")
    run = inner[0]
    floor = min([prof[i][1] for i in run])
    deep = [prof[i][0] for i in run if prof[i][1] < floor + 0.0002]
    target = 0.5 * (min(deep) + max(deep))
    R["dip_mm"] = [round(target * 1e3, 2), round((top - floor) * 1e3, 2)]
    nx = int(round((LAT_UNC + 0.0015) / 0.00035))
    entered = None
    for j in range(2 * nx + 1):
        w = 0.00035 * (j - nx)
        for k in range(5):
            command(lat(target, w, zh))
        for k in range(10):
            command(lat(target, w, floor - 0.0015))
        if tip_local()[2] < floor - 0.0008:
            entered = w
            break
    if entered is None:
        # 3. Forceful re-presses from above inside the dip ("Edge card stalls on the housing top", step 2).
        R["route"].append("represses")
        dip = floor
        t_lo = min([prof[i][0] for i in run]) - PITCH / 2
        t_hi = max([prof[i][0] for i in run]) + PITCH / 2
        mid = 0.5 * (t_lo + t_hi)
        fine = [mid] + [mid + s * k * PITCH / 3 for k in range(1, 8) for s in (1, -1) if t_lo <= mid + s * k * PITCH / 3 <= t_hi]
        WIDE = (0.0005, -0.0005, 0.001, -0.001, 0.0015, -0.0015, 0.002, -0.002)
        tries = [[t, 0.0] for t in fine] + [[mid, w] for w in WIDE]
        for tw in tries:
            if ticks_left() < 150:
                break
            for k in range(4):
                command(lat(tw[0], tw[1], zh))
            for k in range(12):
                command(lat(tw[0], tw[1], DEPTH - 0.008))
            if tip_local()[2] < min(dip, top - 0.0004) - 0.001:
                z0 = tip_local()[2]
                for k in range(20):
                    command(lat(tw[0], tw[1], z0 + (-0.002 - z0) * (k + 1) / 20.0))
                for k in range(20):
                    command(lat(tw[0], tw[1], -0.002))
                cmd = lat(tw[0], tw[1], 0.0)
                m = tip_local()
                if m[2] < 0.0005 and abs(m[0] - cmd[0]) < 0.002 and abs(m[1] - cmd[1]) < 0.002:
                    R["repress_hit_mm"] = [round(tw[0] * 1e3, 2), round(tw[1] * 1e3, 2)]
                    return True
                for k in range(8):
                    command(lat(tw[0], tw[1], zh))
    if entered is None:
        R["failure_stage"] = "wide_pecks_no_catch"
        return False
    R["entered_w_mm"] = round(entered * 1e3, 2)
    z0 = floor - 0.0015
    for k in range(40):
        command(lat(target, entered, z0 + (-0.003 - z0) * (k + 1) / 40.0))
    for k in range(15):
        command(lat(target, entered, -0.003))
    return True


# ---------------------------------------------------------------- tall guides: RGB-D alignment + lift-and-poke descent
def tall_guide_search():
    Z0 = max(START[2], tip_local()[2] + 0.002)
    SPEED = 0.0003
    LAG = 0.0008
    pattern = [[0, 0.0003], [0, -0.0003], [0.0002, 0], [-0.0002, 0], [0, 0.0006], [0, -0.0006],
               [0.0002, 0.0003], [-0.0002, 0.0003], [0.0002, -0.0003], [-0.0002, -0.0003], [0, 0.001], [0, -0.001],
               [0.0004, 0], [-0.0004, 0], [0, 0.0015], [0, -0.0015]]
    stats = {"pokes": 0, "attempts": 0}
    Q = [list(Q0)]

    def descend(off):
        zc = Z0
        while ticks_left() > 40:
            zc -= SPEED
            command([off[0], off[1], zc], Q[0])
            t = tip_local()
            if t[2] < AX_UNC:
                return True, off
            if t[2] - zc > LAG:
                zs = t[2]
                found = None
                for c in pattern:
                    if ticks_left() < 60:
                        break
                    stats["pokes"] += 1
                    x, y = off[0] + c[0], off[1] + c[1]
                    for k in range(3):
                        command([x, y, zs + 0.001], Q[0])
                    for k in range(5):
                        command([x, y, zs - 0.0015], Q[0])
                    if tip_local()[2] < zs - 0.0007:
                        found = [x, y]
                        break
                if found is None:
                    return False, off
                off = found
                zc = tip_local()[2]
        return False, off

    def back_out(off, q):
        t = tip_local()
        for i in range(15):
            a = (i + 1) / 15.0
            command([t[0] + (off[0] - t[0]) * a, t[1] + (off[1] - t[1]) * a, t[2] + (Z0 - t[2]) * a], q)
        for i in range(6):
            command([off[0], off[1], Z0], q)

    # Measure at rest above the guides, nominal orientation, absolute estimate frame.
    OFF[0] = 0.0
    OFF[1] = 0.0
    QC[0] = list(Q0)
    back_out([0.0, 0.0], Q0)
    for k in range(6):
        command([0.0, 0.0, Z0])
    alts = [[0.0, 0.0]]
    va = vision_alignment(tip_abs())
    if va is not None:
        Q[0] = va[0]
        QC[0] = va[0]
        xcmd, ycmd, xs = va[1], va[2], va[3]
        alts = [[xcmd, ycmd]] + [[x, ycmd] for x in xs] + [[0.0, ycmd], [xcmd + 0.0004, ycmd], [xcmd - 0.0004, ycmd]]
        R["vision"] = va[4]
    for off in alts:
        if ticks_left() < 250 and stats["attempts"] > 0:
            break
        stats["attempts"] += 1
        back_out(off, Q[0])
        ok, off2 = descend(off)
        if ok:
            for k in range(20):
                command([off2[0], off2[1], -AX_UNC], Q[0])
            R["tall_off_mm"] = [round(off2[0] * 1e3, 2), round(off2[1] * 1e3, 2)]
            R["tall_stats"] = stats
            return True
    R["tall_stats"] = stats
    R["failure_stage"] = "tall_guide_exhausted"
    return False


# ---------------------------------------------------------------- routing
try:
    if first_z is not None and (first_z < 0.0005 or (LONG_CARD and first_z < AX_UNC + 0.0004)):
        R["route"].append("first_try")
        if hard_stop_seated([0.0, 0.0]):
            R["done"] = True
        else:
            R["route"].append("first_try_rejected")
    if not R.get("done"):
        if high_stall is not None:
            R["route"].append("tall_guide")
            R["done"] = tall_guide_search()
        elif LONG_CARD:
            R["route"].append("edge_card")
            R["done"] = edge_card_search()
        elif EDGE_CARD:
            R["route"].append("recessed_grid")
            f0 = first_z if (first_z is not None and abs(first_z - DEPTH) < 0.0013) else None
            R["done"] = recessed_grid(f0)
        else:
            R["route"].append("compact")
            R["done"] = compact_search()
except Exception as exc:
    R["failure_stage"] = "exception"
    R["error"] = str(exc)[:300]
fin = tip_local()
R["final_local_mm"] = [round(v * 1e3, 2) for v in fin]
R["ticks_left"] = ticks_left()
if not R.get("done") and R["failure_stage"] is None:
    R["failure_stage"] = "not_done"
