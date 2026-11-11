"""Camera-based socket pose estimation for the insertion benchmark.

Enabled with ``INSERTION_SOCKET_SOURCE=vision``: ``InsertionAdapter.get_task_metadata()`` then
reports the pose measured here (and its measured uncertainty) instead of the synthetic
"true pose plus seeded noise" estimate.

What the estimator may use is what a real station has:

* RGB-D frames with intrinsics and extrinsics (``pr_base_link`` / world),
* the task's name and language (to prompt an open-vocabulary detector),
* a coarse prior: the plug tip pose the arm was brought to (``end_effector_poses``) minus the
  nominal standoff, the seated depth, and the plug bounds,

and never the scene's socket geometry or the truth sites.

Pipeline (one "look" before the episode starts):

1. **Look views.** Four 1280 x 960 RGB-D views aimed at the centre of the plug's footprint over
   the prior mouth, framed to the plug size, with the arm and held plug out of view: the
   station's socket scan before the arm moves in (equivalently a parked-arm look phase). View 0
   is 5 deg off the prior axis, tilted along the plug's long axis so it looks down a slot's
   length (slot floor, key); three oblique views at 35 deg (15-20 deg, staying above the board,
   for sideways-facing sockets).
2. **Detection + segmentation** (``insertion_skill/perception/vision_server.py``, a GPU service):
   Grounding DINO and OWLv2 with task-specific prompts; the box that contains the projected
   prior and fits its gate wins; SAM 2 masks it. The ROI is that box unioned with the plug
   footprint around the prior (a tight box must not truncate the opening).
3. **Face planes.** Height-histogram peaks of the ROI points facing the prior axis, each resolved
   by RANSAC in its height slab (normal within 8 deg of the axis) and refined by least squares.
4. **Aperture.** On each plane, every pixel whose depth lands > 0.8 mm behind it is a ray that
   passed through an opening: where it crossed the plane marks the aperture. Openings are
   connected aperture cells ringed by face/solid cells that some camera saw. Collinear pieces
   merge into one slot (a flush key splits it); similar compact openings pair up (two-pin
   plugs). A second "deep" family (> 2 mm behind, shallow recesses count as wall) covers lipped
   connectors whose rim is not one plane. The hypothesis explaining the most aperture near the
   prior wins; an opening nested inside it on a lower plane (a connector in a pocket) replaces it.
5. **Pose.** Insertion axis = plane normal. Mouth = opening centre on the plane, or for a slot
   the key: an interior stretch the near-axis view sees as shallow/flush between deep floor.
   Yaw = the opening axis (or pin-pair line) resolved against the prior's in-plane axes. Seat =
   mouth - seated_depth * axis, or the floor seen through the opening when no rim plane
   encloses it or the floor sits above that seat.
6. **Checks and uncertainty.** Leave-one-view-out re-estimates (multi-view consistency), plane
   residual, floor spread, enclosure and detector agreement. Bounds = a per-regime floor fitted
   against truth on calibration seeds 101-130 + the measured spread (x2 if the views disagree,
   x2 if no detector confirmed the socket). If the checks fail, the views are re-rendered from a
   second viewpoint set (re-detection); if that fails too, the estimate is the prior with
   prior-sized bounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import io
import json
import math
import os
import time
from typing import Any
import urllib.request

import numpy as np

#: Task-specific detector prompts (a real cell would know which socket type it is plugging into).
SOCKET_PROMPTS: dict[str, list[str]] = {
    "adapter_into_wall_socket": ["wall power socket", "electrical outlet"],
    "nvme_into_m2_dock": ["slot in the dock", "docking station"],
    "nvme_into_m2_socket": ["M.2 connector", "black connector"],
    "pcie_card_into_x16_slot": ["PCIe slot", "black slot connector"],
    "pcie_riser_into_x16_slot": ["PCIe slot", "black slot connector"],
    "ram_into_dimm_slot": ["DIMM slot", "memory slot"],
    "usb_a_into_receptacle": ["USB port", "panel"],
    "usb_c_into_receptacle": ["USB-C port", "panel"],
}
DEFAULT_PROMPTS = ["socket", "slot", "port"]

#: Fallback (prior-only) bounds: what the arm's start pose alone tells you. Axially the arm may
#: start backed off over tall guides (~11 mm above the DIMM slot), hence 15 mm.
PRIOR_BOUNDS = {"lateral_m": 0.005, "axial_m": 0.015, "tilt_rad": math.radians(3.0), "yaw_rad": math.radians(3.0)}

LOOK_WIDTH, LOOK_HEIGHT = 1280, 960
CELL_M = 0.0001
FACE_TOL_M = 0.0003
THROUGH_TOL_M = 0.0008
DEEP_TOL_M = 0.002
PLANE_MAX_TILT_RAD = math.radians(8.0)
PRIOR_GATE_M = 0.008
#: An opening shorter than this fraction of the plug's long extent cannot be the socket (the
#: smallest true ratio is 0.45: a wall socket's pin spacing against the adapter body).
MIN_OPENING_FRACTION = 0.35


# --------------------------------------------------------------------------- geometry helpers
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _wrap(angle: float, period: float) -> float:
    return (angle + 0.5 * period) % period - 0.5 * period


@dataclass
class View:
    name: str
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    world_from_cam: np.ndarray  # OpenCV axes: x right, y down, z forward
    detection: dict = field(default_factory=dict)

    def project(self, points: np.ndarray) -> np.ndarray:
        cam = (np.linalg.inv(self.world_from_cam) @ np.c_[points, np.ones(len(points))].T)[:3]
        uv = (self.K @ cam)[:2] / cam[2]
        return uv.T

    def rays(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(points, unit ray directions) in world for pixel (row, col) indices with depth."""
        rows, cols = pixels
        z = self.depth[rows, cols]
        x = (cols + 0.5 - self.K[0, 2]) / self.K[0, 0]
        y = (rows + 0.5 - self.K[1, 2]) / self.K[1, 1]
        cam = np.stack([x * z, y * z, z], axis=1)
        R, t = self.world_from_cam[:3, :3], self.world_from_cam[:3, 3]
        points = cam @ R.T + t
        dirs = np.stack([x, y, np.ones_like(x)], axis=1) @ R.T
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return points, dirs


@dataclass
class Prior:
    mouth: np.ndarray  # world
    rotation: np.ndarray  # world, columns x, y, z (z out of the socket)
    window_m: float  # half size of the search window on the face plane
    plug_extent: np.ndarray  # lateral full extents (x, y) of the plug
    plug_min: np.ndarray = field(default_factory=lambda: np.zeros(2))  # lateral plug bounds in the tip frame
    plug_max: np.ndarray = field(default_factory=lambda: np.zeros(2))
    look_centre: np.ndarray | None = None  # centre of the plug footprint over the socket (camera aim)
    view_half_m: float = 0.0  # half size of the footprint the look views must frame


# --------------------------------------------------------------------------- vision client
class VisionClient:
    """HTTP client for perception/vision_server.py (Grounding DINO + OWLv2 + SAM 2)."""

    def __init__(self, url: str | None = None, timeout_s: float = 60.0) -> None:
        self.url = (url or os.environ.get("INSERTION_VISION_URL", "http://127.0.0.1:8765")).rstrip("/")
        self.timeout_s = timeout_s

    def detect(self, images: list[np.ndarray], prompts: list[str], rois: list | None = None) -> list[dict]:
        request = {"prompts": prompts, "n_images": len(images), "roi": rois, "box_threshold": 0.15,
                   "text_threshold": 0.15, "owl_threshold": 0.05, "max_boxes": 4}
        buffer = io.BytesIO()
        np.savez(buffer, request=np.array(json.dumps(request)),
                 **{f"image_{i}": np.ascontiguousarray(image) for i, image in enumerate(images)})
        http = urllib.request.Request(self.url + "/detect", data=buffer.getvalue(), method="POST",
                                      headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(http, timeout=self.timeout_s) as response:
            reply = np.load(io.BytesIO(response.read()), allow_pickle=False)
        results = []
        for i, image in enumerate(images):
            masks = np.unpackbits(reply[f"masks_{i}"], axis=-1, count=image.shape[1]).astype(bool)
            results.append({
                "boxes": reply[f"boxes_{i}"], "scores": reply[f"scores_{i}"],
                "labels": json.loads(str(reply[f"labels_{i}"])),
                "detector": json.loads(str(reply[f"detector_{i}"])),
                "masks": masks, "mask_scores": reply[f"mask_scores_{i}"], "time_s": float(reply[f"time_{i}"]),
            })
        return results


# --------------------------------------------------------------------------- prior and views
def coarse_prior(adapter: Any) -> Prior:
    """Socket mouth prior from where the arm was brought to (nominal tip pose minus the standoff)."""
    env = adapter.native_env
    data = env.data
    from cap_harness.insertion.adapter import _transform

    world_from_tip = _transform(data.xmat[adapter._ee_id], data.xpos[adapter._ee_id]) @ adapter._ee_from_tip_nominal
    standoff = float(env.manifest.get("standoff_m", 0.01))
    extent = adapter._plug_extent_in_tip()
    lateral = np.array([extent[3] - extent[0], extent[4] - extent[1]])
    rotation = world_from_tip[:3, :3].copy()
    mouth = world_from_tip[:3, 3] - rotation[:, 2] * standoff
    lo, hi = np.array(extent[:2]), np.array(extent[3:5])
    # The tip frame need not be centred in the plug (keyed edge cards): the search window must
    # reach the far end of the footprint, and the cameras aim at the footprint centre.
    reach = float(np.abs(np.r_[lo, hi]).max())
    centre = mouth + rotation[:, :2] @ (0.5 * (lo + hi))
    return Prior(mouth=mouth, rotation=rotation, window_m=reach + PRIOR_GATE_M + 0.004,
                 plug_extent=lateral, plug_min=lo, plug_max=hi, look_centre=centre,
                 view_half_m=0.5 * float(lateral.max()) + PRIOR_GATE_M)


def _view_directions(prior: Prior, variant: int) -> list[tuple[str, np.ndarray]]:
    """Unit directions (socket -> camera). View 0 is nearly on-axis, tilted along the plug's long
    axis so it looks down a slot's length (it sees the slot floor and any key); the others are
    oblique. Horizontal-axis sockets keep every camera above the surrounding surface."""
    z = prior.rotation[:, 2]
    long_axis = prior.rotation[:, 0] if prior.plug_extent[0] >= prior.plug_extent[1] else prior.rotation[:, 1]
    world_up = np.array([0.0, 0.0, 1.0])
    up = world_up - z * (z @ world_up)
    vertical = np.linalg.norm(up) < 0.3
    if vertical:  # vertical insertion axis: reference the plug's axes
        up = np.cross(z, long_axis)
        oblique = [(35, 90), (35, 210), (35, 330)] if variant == 0 else [(42, 30), (42, 150), (42, 270)]
    else:
        # Horizontal-ish axis (a connector facing sideways): stay low, the opening is a thin slit.
        oblique = [(20, 30), (15, 90), (20, 150)] if variant == 0 else [(28, 60), (25, 90), (28, 120)]
    up = _unit(up)
    side = np.cross(up, z)
    near_tilt = math.radians(5.0 if variant == 0 else 9.0)
    sign = 1.0 if variant == 0 else -1.0
    directions = [_unit(math.cos(near_tilt) * z + sign * math.sin(near_tilt) * _unit(long_axis - z * (z @ long_axis)))]
    for off, az in oblique:
        o, a = math.radians(off), math.radians(az)
        directions.append(_unit(math.cos(o) * z + math.sin(o) * (math.cos(a) * side + math.sin(a) * up)))
    return [(f"look{variant}_{k}", d) for k, d in enumerate(directions)]


def render_look_views(adapter: Any, prior: Prior, variant: int = 0) -> list[View]:
    """RGB-D views of the prior socket mouth with the arm and held plug out of view."""
    import mujoco

    env = adapter.native_env
    model, data = env.model, env.data
    fovy = float(model.vis.global_.fovy)
    width = min(LOOK_WIDTH, int(model.vis.global_.offwidth))
    height = min(LOOK_HEIGHT, int(model.vis.global_.offheight))
    distance = float(np.clip(1.3 * prior.view_half_m / math.tan(math.radians(fovy) / 2), 0.07, 0.45))
    aim = prior.look_centre if prior.look_centre is not None else prior.mouth
    hidden = [g for g in range(model.ngeom)
              if model.body(model.geom_bodyid[g]).name.startswith("pr_right") or model.body(model.geom_bodyid[g]).name == "plug"]
    saved = model.geom_group[hidden].copy()
    model.geom_group[hidden] = 5
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[5] = 0
    renderer = mujoco.Renderer(model, height, width)
    views = []
    try:
        for name, direction in _view_directions(prior, variant):
            forward = -direction
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = aim
            camera.distance = distance
            camera.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
            camera.elevation = math.degrees(math.asin(np.clip(forward[2], -1.0, 1.0)))
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=camera, scene_option=options)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=camera, scene_option=options)
            depth = np.asarray(renderer.render(), dtype=np.float64).copy()
            renderer.disable_depth_rendering()
            right = _unit(np.cross(forward, [0.0, 0.0, 1.0]))
            cam_up = np.cross(right, forward)
            world_from_cam = np.eye(4)
            world_from_cam[:3, :3] = np.column_stack([right, -cam_up, forward])  # OpenCV axes
            world_from_cam[:3, 3] = aim - forward * distance
            focal = 0.5 * height / math.tan(math.radians(fovy) / 2)
            K = np.array([[focal, 0, 0.5 * width], [0, focal, 0.5 * height], [0, 0, 1.0]])
            far = depth.max()
            depth[depth >= far * 0.999] = 0.0  # background / far plane: no return
            views.append(View(name, rgb, depth, K, world_from_cam))
    finally:
        renderer.close()
        model.geom_group[hidden] = saved
    return views


def add_sensor_noise(views: list[View], rng: np.random.Generator, depth_sigma_m: float, extrinsic_m: float) -> None:
    """Optional realism: depth noise (sigma at 0.15 m, growing with z^2) and extrinsic calibration error."""
    for view in views:
        if depth_sigma_m > 0:
            valid = view.depth > 0
            sigma = depth_sigma_m * (view.depth / 0.15) ** 2
            view.depth = np.where(valid, view.depth + rng.normal(size=view.depth.shape) * sigma, 0.0)
        if extrinsic_m > 0:
            view.world_from_cam = view.world_from_cam.copy()
            view.world_from_cam[:3, 3] += rng.normal(size=3) * extrinsic_m


# --------------------------------------------------------------------------- 2-D detection
def _prior_roi(view: View, prior: Prior, margin_m: float) -> np.ndarray:
    x, y = prior.rotation[:, 0], prior.rotation[:, 1]
    corners = [prior.mouth + sx * margin_m * x + sy * margin_m * y for sx in (-1, 1) for sy in (-1, 1)]
    uv = view.project(np.asarray(corners))
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def _footprint_roi(view: View, prior: Prior, margin_m: float) -> np.ndarray:
    """Image box of the plug's lateral footprint around the prior mouth, plus a margin."""
    x, y = prior.rotation[:, 0], prior.rotation[:, 1]
    corners = [prior.mouth + (prior.plug_min[0] - margin_m if sx < 0 else prior.plug_max[0] + margin_m) * x
               + (prior.plug_min[1] - margin_m if sy < 0 else prior.plug_max[1] + margin_m) * y
               for sx in (-1, 1) for sy in (-1, 1)]
    uv = view.project(np.asarray(corners))
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def choose_detection(view: View, result: dict, prior: Prior) -> dict:
    """Pick the detection consistent with the projected prior; None-box means geometry-only ROI."""
    centre = view.project(prior.mouth[None])[0]
    gate = _prior_roi(view, prior, PRIOR_GATE_M + 0.5 * float(prior.plug_extent.max()))
    best, best_score = None, 0.0
    h, w = view.depth.shape
    for k, box in enumerate(result["boxes"]):
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0) / (h * w)
        contains = x0 - 5 <= centre[0] <= x1 + 5 and y0 - 5 <= centre[1] <= y1 + 5
        if not contains or area > 0.9:
            continue
        # Overlap with the prior gate; prefer boxes that fit the gate (not the whole image).
        ix = max(0.0, min(x1, gate[2]) - max(x0, gate[0]))
        iy = max(0.0, min(y1, gate[3]) - max(y0, gate[1]))
        overlap = ix * iy / max((x1 - x0) * (y1 - y0), 1.0)
        score = float(result["scores"][k]) * (0.5 + 0.5 * overlap)
        if score > best_score:
            best, best_score = k, score
    roi = _prior_roi(view, prior, prior.window_m)
    choice = {"index": best, "score": best_score, "roi": roi, "mask": None}
    if best is not None:
        box = result["boxes"][best]
        pad = 0.15 * max(box[2] - box[0], box[3] - box[1]) + 10
        choice.update(box=box, label=result["labels"][best], detector=result["detector"][best],
                      mask=result["masks"][best], mask_score=float(result["mask_scores"][best]))
        # ROI = (padded detection box U plug footprint around the prior) clipped to the prior window:
        # the box removes clutter, the footprint keeps a tight or partial box from truncating the opening.
        foot = _footprint_roi(view, prior, 0.004)
        union = np.array([min(box[0] - pad, foot[0]), min(box[1] - pad, foot[1]),
                          max(box[2] + pad, foot[2]), max(box[3] + pad, foot[3])])
        choice["roi"] = np.array([max(roi[0], union[0]), max(roi[1], union[1]),
                                  min(roi[2], union[2]), min(roi[3], union[3])])
    return choice


# --------------------------------------------------------------------------- 3-D fusion
def _roi_pixels(view: View, roi: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    h, w = view.depth.shape
    x0, y0 = max(int(roi[0]), 0), max(int(roi[1]), 0)
    x1, y1 = min(int(math.ceil(roi[2])), w), min(int(math.ceil(roi[3])), h)
    rows, cols = np.mgrid[y0:y1:stride, x0:x1:stride]
    rows, cols = rows.ravel(), cols.ravel()
    keep = view.depth[rows, cols] > 0
    return rows[keep], cols[keep]


def pixel_normals(view: View, rows: np.ndarray, cols: np.ndarray, step: int = 2) -> np.ndarray:
    """Surface normals (world) at the given pixels from depth differences; NaN where undefined."""
    h, w = view.depth.shape
    r0, r1 = np.clip(rows - step, 0, h - 1), np.clip(rows + step, 0, h - 1)
    c0, c1 = np.clip(cols - step, 0, w - 1), np.clip(cols + step, 0, w - 1)
    pa, _ = view.rays((rows, c0))
    pb, _ = view.rays((rows, c1))
    pc, _ = view.rays((r0, cols))
    pd, _ = view.rays((r1, cols))
    n = np.cross(pb - pa, pd - pc)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    valid = (view.depth[rows, c0] > 0) & (view.depth[rows, c1] > 0) & (view.depth[r0, cols] > 0) & (view.depth[r1, cols] > 0)
    n = n / np.where(norm > 0, norm, 1.0)
    n[~valid] = np.nan
    return n


def fit_planes(points: np.ndarray, axis: np.ndarray, rng: np.random.Generator, max_planes: int = 6) -> list[dict]:
    """Planes facing along ``axis``: coarse peaks of the height histogram (the prior axis may be
    a degree off, which smears a long plane over ~2 mm of height), each resolved by RANSAC inside
    its height slab (normal within PLANE_MAX_TILT_RAD of the axis) and refined by least squares."""
    if len(points) < 50:
        return []
    heights = points @ axis
    bin_m = 0.00025
    lo = heights.min()
    hist = np.bincount(((heights - lo) / bin_m).astype(int))
    smooth = np.convolve(hist, np.ones(5), mode="same")
    min_inliers = max(150, int(0.003 * len(points)))
    peaks: list[int] = []  # greedy non-maximum suppression, 1 mm apart
    for k in np.argsort(-smooth):
        if smooth[k] < min_inliers or len(peaks) >= 2 * max_planes:
            break
        if all(abs(int(k) - q) * bin_m >= 0.001 for q in peaks):
            peaks.append(int(k))
    planes = []
    cos_max = math.cos(PLANE_MAX_TILT_RAD)
    for k in peaks:
        h0 = lo + (k + 0.5) * bin_m
        slab = points[np.abs(heights - h0) < 0.0015]
        if len(slab) < min_inliers:
            continue
        sample = slab if len(slab) <= 20000 else slab[rng.choice(len(slab), 20000, replace=False)]
        best_count, best = 0, None
        for _ in range(150):
            p = sample[rng.choice(len(sample), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n /= norm
            if abs(n @ axis) < cos_max:
                continue
            count = int(np.count_nonzero(np.abs((sample - p[0]) @ n) < FACE_TOL_M))
            if count > best_count:
                best_count, best = count, (n if n @ axis > 0 else -n, p[0])
        if best is None:
            continue
        n, p0 = best
        inl = points[np.abs((points - p0) @ n) < FACE_TOL_M]
        for _ in range(3):  # least-squares refinement on the inliers
            if len(inl) < 3:
                break
            centroid = inl.mean(axis=0)
            _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
            n_new = vt[2] if vt[2] @ axis > 0 else -vt[2]
            if n_new @ axis < cos_max:
                break
            n, p0 = n_new, centroid
            inl = points[np.abs((points - p0) @ n) < FACE_TOL_M]
        if len(inl) < min_inliers:
            continue
        if any(abs((p0 - q["point"]) @ q["normal"]) < 2 * FACE_TOL_M and q["normal"] @ n > math.cos(math.radians(1))
               for q in planes):
            continue  # duplicate
        residual = (inl - p0) @ n
        planes.append({"normal": n, "point": p0, "inliers": len(inl), "rms_m": float(np.sqrt(np.mean(residual**2)))})
        if len(planes) >= max_planes:
            break
    return planes


@dataclass
class Grid:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    half: float
    cell: float

    @property
    def size(self) -> int:
        return int(round(2 * self.half / self.cell))

    def flat_index(self, points: np.ndarray) -> np.ndarray:
        rel = points - self.origin
        i = np.floor((rel @ self.u + self.half) / self.cell).astype(np.int64)
        j = np.floor((rel @ self.v + self.half) / self.cell).astype(np.int64)
        ok = (i >= 0) & (i < self.size) & (j >= 0) & (j < self.size)
        return (i * self.size + j)[ok]

    def count(self, points: np.ndarray) -> np.ndarray:
        return np.bincount(self.flat_index(points), minlength=self.size**2).reshape(self.size, self.size)

    def uv(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return np.stack([(i + 0.5) * self.cell - self.half, (j + 0.5) * self.cell - self.half], axis=-1)


def classify(view_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]], grid: Grid) -> dict[str, np.ndarray]:
    """Plane-grid evidence from (points, ray dirs, camera centre) per view.

    face: hits on the plane; solid: hits in front of it (projected down); through: rays that
    crossed the plane and hit something more than THROUGH_TOL_M behind it, binned where they
    crossed; deep: the same for hits more than DEEP_TOL_M behind (slot floors, not key tops).
    """
    size = grid.size
    out = {k: np.zeros((size, size), np.int32) for k in ("face", "solid", "through", "deep")}
    for points, dirs, cam in view_data:
        s = (points - grid.origin) @ grid.n
        out["face"] += grid.count(points[np.abs(s) < FACE_TOL_M])
        front = s > THROUGH_TOL_M
        out["solid"] += grid.count(points[front] - np.outer(s[front], grid.n))
        behind = s < -THROUGH_TOL_M
        denom = dirs[behind] @ grid.n
        t = ((grid.origin - cam) @ grid.n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        crossing = cam + dirs[behind] * t[:, None]
        out["through"] += grid.count(crossing)
        out["deep"] += grid.count(crossing[s[behind] < -DEEP_TOL_M])
    return out


def find_holes(evidence: dict[str, np.ndarray], grid: Grid, family: str = "through") -> list[dict]:
    """Enclosed openings: connected open cells ringed by wall cells, off the window edge.

    family "through": open = rays passed more than THROUGH_TOL_M behind the plane; walls = face
    or solid. family "deep": open = more than DEEP_TOL_M behind (a slot floor), walls also include
    shallow recesses (chamfers, key tops) -- for sockets whose rim is not one plane (M.2 lips).
    """
    from scipy import ndimage

    closing = max(1, int(round(0.0002 / grid.cell)))
    blocking = evidence["face"] + evidence["solid"]
    if family == "deep":
        opened = evidence["deep"]
        blocking = blocking + (evidence["through"] - evidence["deep"])
    else:
        opened = evidence["through"]
    through = (opened > 0) & (opened >= evidence["face"])
    # Close sub-cell sampling gaps (pixels are larger than cells far from the camera).
    walled = blocking > opened
    through = ndimage.binary_closing(through, iterations=closing) & ~walled  # never bridge a seen wall
    wall = ndimage.binary_closing(blocking > 0, iterations=closing) & ~through
    # Cells no ray reached (after closing the sub-pixel sampling gaps).
    unknown = ~ndimage.binary_closing((evidence["face"] + evidence["solid"] + evidence["through"]) > 0,
                                      iterations=closing + 1)
    labels, _ = ndimage.label(through)
    holes = []
    size = grid.size
    ring_width = max(2, int(round(0.0004 / grid.cell)))
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == size or sl[1].stop == size:
            continue  # touches the window edge: exterior, not an enclosed opening
        region = labels[sl] == k
        area = region.sum() * grid.cell**2
        if area < 0.3e-6:
            continue
        pad = tuple(slice(max(s.start - ring_width, 0), min(s.stop + ring_width, size)) for s in sl)
        local = labels[pad] == k
        ring = ndimage.binary_dilation(local, iterations=ring_width) & ~local
        enclosure = float(wall[pad][ring].mean()) if ring.any() else 0.0
        unseen = float(unknown[pad][ring].mean()) if ring.any() else 1.0
        if enclosure < 0.6 or unseen > 0.1:
            continue  # open to the exterior, or bounded by cells no camera saw (ROI edge)
        fill = float(region.sum() / max(ndimage.binary_fill_holes(region).sum(), 1))
        ii, jj = np.nonzero(region)
        ii, jj = ii + sl[0].start, jj + sl[1].start
        coords = grid.uv(ii, jj)
        centre = coords.mean(axis=0)
        cov = np.cov((coords - centre).T) if len(coords) > 2 else np.eye(2) * 1e-12
        evals, evecs = np.linalg.eigh(cov)
        holes.append({
            "centre_uv": centre, "area_m2": float(area), "enclosure": enclosure, "fill": fill,
            "major_angle": float(math.atan2(evecs[1, 1], evecs[0, 1])),
            "aspect": float(math.sqrt(max(evals[1], 1e-18) / max(evals[0], 1e-18))),
            "extent_m": float(math.sqrt(12 * max(evals[1], 0.0))),
            "cells": (ii, jj),
        })
    return holes


def _collinear(a: dict, b: dict) -> bool:
    """Two components of one key-split slot: same axis line, small end gap."""
    big, small = (a, b) if a["area_m2"] >= b["area_m2"] else (b, a)
    if big["aspect"] < 2.0:
        return False
    axis = np.array([math.cos(big["major_angle"]), math.sin(big["major_angle"])])
    normal = np.array([-axis[1], axis[0]])
    d = small["centre_uv"] - big["centre_uv"]
    if abs(d @ normal) > 0.0006:
        return False
    gap = abs(d @ axis) - 0.5 * big["extent_m"] - 0.5 * small["extent_m"]
    return gap < 0.004


def _group_shape(holes: list[dict], members: list[int], grid: Grid) -> dict:
    ii = np.concatenate([holes[k]["cells"][0] for k in members])
    jj = np.concatenate([holes[k]["cells"][1] for k in members])
    coords = grid.uv(ii, jj)
    mean = coords.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((coords - mean).T) if len(coords) > 2 else np.eye(2))
    axis = evecs[:, 1]
    along = (coords - mean) @ axis
    # Slot centre: midpoint of the extent along the major axis (robust to a key gap), mean across.
    centre = mean + axis * 0.5 * (along.min() + along.max())
    return {"centre_uv": centre, "angle": float(math.atan2(axis[1], axis[0])),
            "aspect": float(math.sqrt(max(evals[1], 1e-18) / max(evals[0], 1e-18))),
            "length_m": float(along.max() - along.min()),
            "width_m": float(math.sqrt(12 * max(evals[0], 0.0))), "kind": "opening"}


def hypotheses(holes: list[dict], grid: Grid) -> list[dict]:
    """Openings (collinear components merged into one slot) and pairs of similar openings (two pins)."""
    parent = list(range(len(holes)))

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k

    for a in range(len(holes)):
        for b in range(a + 1, len(holes)):
            if _collinear(holes[a], holes[b]):
                parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for k in range(len(holes)):
        groups.setdefault(root(k), []).append(k)
    openings = []
    for members in groups.values():
        shape = _group_shape(holes, members, grid)
        shape.update(members=members, area_m2=float(sum(holes[k]["area_m2"] for k in members)))
        openings.append(shape)
    result = list(openings)
    for a in range(len(openings)):
        for b in range(a + 1, len(openings)):
            ratio = openings[a]["area_m2"] / openings[b]["area_m2"]
            compact = openings[a]["aspect"] < 3.0 and openings[b]["aspect"] < 3.0
            if compact and 0.6 < ratio < 1.67:
                d = openings[b]["centre_uv"] - openings[a]["centre_uv"]
                result.append({"members": openings[a]["members"] + openings[b]["members"],
                               "centre_uv": 0.5 * (openings[a]["centre_uv"] + openings[b]["centre_uv"]),
                               "angle": float(math.atan2(d[1], d[0])), "aspect": 10.0, "kind": "pair",
                               "area_m2": openings[a]["area_m2"] + openings[b]["area_m2"],
                               "length_m": float(np.linalg.norm(d)), "width_m": 0.0})
    return result


def find_key(hyp: dict, near: dict[str, np.ndarray], grid: Grid, prior_along: float) -> dict | None:
    """Locate a slot's key: an interior stretch of the slot that the near-axis view sees as
    shallow (key top) or flush, with the deep slot floor on both sides of it."""
    from scipy import ndimage

    if hyp["kind"] != "opening" or hyp["aspect"] < 5.0:
        return None
    axis = np.array([math.cos(hyp["angle"]), math.sin(hyp["angle"])])
    normal = np.array([-axis[1], axis[0]])
    band = max(0.35 * hyp["width_m"], grid.cell)
    half_len = 0.5 * hyp["length_m"]
    nbins = int(math.ceil(2 * half_len / grid.cell)) + 1

    def profile(mask: np.ndarray) -> np.ndarray:
        ii, jj = np.nonzero(mask)
        rel = grid.uv(ii, jj) - hyp["centre_uv"]
        keep = (np.abs(rel @ normal) < band) & (np.abs(rel @ axis) <= half_len)
        bins = np.floor((rel[keep] @ axis + half_len) / grid.cell).astype(int).clip(0, nbins - 1)
        return np.bincount(bins, minlength=nbins) > 0

    deep = ndimage.binary_closing(profile(near["deep"] > 0), iterations=max(1, int(round(0.0003 / grid.cell))))
    seen = profile((near["face"] + near["through"] + near["solid"]) > 0)
    occupied = np.nonzero(deep)[0]
    if len(occupied) < 2:
        return None
    labels, count = ndimage.label(~deep)
    keys = []
    for k in range(1, count + 1):
        idx = np.nonzero(labels == k)[0]
        if idx[0] <= occupied[0] or idx[-1] >= occupied[-1]:
            continue  # not interior: slot ends
        length = len(idx) * grid.cell
        if not (0.0004 <= length <= 0.0035) or seen[idx].mean() < 0.5:
            continue
        # Deep floor must continue for >= 1.5 mm on both sides.
        left = deep[max(idx[0] - int(0.0015 / grid.cell), 0):idx[0]].mean() if idx[0] > 0 else 0.0
        right = deep[idx[-1] + 1:idx[-1] + 1 + int(0.0015 / grid.cell)].mean()
        if left < 0.6 or right < 0.6:
            continue
        along = (0.5 * (idx[0] + idx[-1]) + 0.5) * grid.cell - half_len
        keys.append({"along_m": along, "length_m": length})
    if not keys:
        return None
    key = min(keys, key=lambda k: abs(k["along_m"] - prior_along))
    key["candidates"] = len(keys)
    key["centre_uv"] = hyp["centre_uv"] + axis * key["along_m"]
    return key


def slot_floor(cells: tuple[np.ndarray, np.ndarray], view_data: list, grid: Grid) -> dict | None:
    """Height (along the plane normal) of the floor seen through an opening: the mode of the
    deep hits whose rays crossed the plane inside the opening. For slots the floor is the seat."""
    mask = np.zeros((grid.size, grid.size), bool)
    mask[cells] = True
    heights = []
    for points, dirs, cam in view_data:
        if len(points) == 0:
            continue
        s = (points - grid.origin) @ grid.n
        deep = s < -DEEP_TOL_M
        denom = dirs[deep] @ grid.n
        t = ((grid.origin - cam) @ grid.n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        rel = cam + dirs[deep] * t[:, None] - grid.origin
        i = np.floor((rel @ grid.u + grid.half) / grid.cell).astype(np.int64)
        j = np.floor((rel @ grid.v + grid.half) / grid.cell).astype(np.int64)
        ok = (i >= 0) & (i < grid.size) & (j >= 0) & (j < grid.size)
        inside = np.zeros(len(i), bool)
        inside[ok] = mask[i[ok], j[ok]]
        heights.append(s[deep][inside])
    if not heights:
        return None
    h = np.concatenate(heights)
    if len(h) < 50:
        return None
    bin_m = 0.0001
    hist = np.bincount(((h - h.min()) / bin_m).astype(int))
    smooth = np.convolve(hist, np.ones(3), mode="same")
    peak = h.min() + (np.argmax(smooth) + 0.5) * bin_m
    near = h[np.abs(h - peak) < 0.0003]
    return {"height_m": float(np.median(near)), "count": int(len(near)), "fraction": float(len(near) / len(h)),
            "mad_m": float(np.median(np.abs(near - np.median(near))))}


def analyse_plane(plane: dict, view_data: list, prior: Prior) -> dict | None:
    n = plane["normal"]
    origin = prior.mouth + n * ((plane["point"] - prior.mouth) @ n)
    u = _unit(prior.rotation[:, 0] - n * (prior.rotation[:, 0] @ n))
    v = np.cross(n, u)
    cell = max(CELL_M, 2 * prior.window_m / 1200)
    grid = Grid(origin=origin, u=u, v=v, n=n, half=prior.window_m, cell=cell)
    evidence = classify(view_data, grid)
    near = classify(view_data[:1], grid)
    results = []
    for family in ("through", "deep"):
        holes = find_holes(evidence, grid, family)
        if not holes:
            continue
        candidates = hypotheses(holes, grid)
        for c in candidates:
            c["key"] = None
            if c["kind"] == "opening" and c["aspect"] >= 5.0:
                axis = np.array([math.cos(c["angle"]), math.sin(c["angle"])])
                c["key"] = find_key(c, near, grid, float(-c["centre_uv"] @ axis))
            c["target_uv"] = c["key"]["centre_uv"] if c["key"] is not None else c["centre_uv"]
            c["distance_m"] = float(np.linalg.norm(c["target_uv"]))
        # Near the prior, and big enough for the plug: an opening (or pin spacing) far shorter than
        # the plug's footprint is some other feature (seen: a 3 mm recess beside an M.2 slot).
        candidates = [c for c in candidates if c["distance_m"] < PRIOR_GATE_M
                      and c["length_m"] >= MIN_OPENING_FRACTION * float(prior.plug_extent.max())]
        if not candidates:
            continue
        # The hypothesis explaining the most aperture wins (fragments lose to the slot they belong to).
        best = max(candidates, key=lambda c: (c["area_m2"], -c["distance_m"]))
        best["enclosure"] = float(np.mean([holes[k]["enclosure"] for k in best["members"]]))
        best["fill"] = float(min(holes[k]["fill"] for k in best["members"]))
        best["family"] = family
        best["cells"] = tuple(np.concatenate([holes[k]["cells"][d] for k in best["members"]]) for d in (0, 1))
        best["floor"] = slot_floor(best["cells"], view_data, grid) if best["aspect"] >= 5.0 and best["kind"] == "opening" else None
        results.append({"grid": grid, "holes": holes, "best": best, "evidence": evidence, "plane": plane})
    if not results:
        return None
    # A deep-floor opening only stands in when the rim itself does not enclose a comparable one.
    return max(results, key=lambda r: (r["best"]["area_m2"] * (1.0 if r["best"]["family"] == "through" else 0.5)))


def _yaw_correction(angle: float, aspect: float) -> tuple[float, bool]:
    """In-plane rotation of the prior x axis onto the measured opening axis (mod 90 deg)."""
    if aspect < 1.4:
        return 0.0, False
    return _wrap(angle, math.pi / 2), True


def estimate_from_views(views: list[View], prior: Prior, seated_depth: float, rng: np.random.Generator) -> dict:
    """Fuse the views into a seated-tip pose (world) plus diagnostics; raises ValueError on failure."""
    view_data, plane_points = [], []
    z = prior.rotation[:, 2]
    for view in views:
        rows, cols = _roi_pixels(view, view.detection["roi"])
        if len(rows) == 0:
            view_data.append((np.zeros((0, 3)), np.zeros((0, 3)), view.world_from_cam[:3, 3]))
            continue
        points, dirs = view.rays((rows, cols))
        rel = points - prior.mouth
        inside = np.linalg.norm(rel - np.outer(rel @ z, z), axis=1) < prior.window_m
        points, dirs, rows, cols = points[inside], dirs[inside], rows[inside], cols[inside]
        view_data.append((points, dirs, view.world_from_cam[:3, 3]))
        # Face-plane candidates come from the whole ROI (the SAM mask often covers only the
        # opening itself, while the face is the rim around it), keeping surfaces facing the prior axis.
        # (only side walls are dropped: narrow rims have noisy normals at their edges).
        normals = pixel_normals(view, rows, cols, step=1)
        facing = np.abs(normals @ z) > math.cos(math.radians(30.0))
        plane_points.append(points[facing])
    pts = np.concatenate(plane_points) if plane_points else np.zeros((0, 3))
    if len(pts) < 200:
        raise ValueError("no depth inside the regions of interest")
    planes = fit_planes(pts, z, rng)
    if not planes:
        raise ValueError("no face plane")
    analyses = [a for a in (analyse_plane(p, view_data, prior) for p in planes) if a is not None]
    if not analyses:
        raise ValueError("no enclosed opening near the prior")
    # Start from the best-enclosed opening nearest the prior; then descend into nested openings:
    # an opening on a lower plane lying inside the current one (a connector inside a pocket)
    # replaces it, unless it is a ring around a recessed tongue (low fill ratio).
    def score(a: dict) -> float:
        best = a["best"]
        slot_without_key = best["kind"] == "opening" and best["aspect"] >= 5.0 and best.get("key") is None
        return (best["area_m2"] * (0.5 if best["family"] == "deep" else 1.0) * (0.3 if slot_without_key else 1.0)
                * (1.0 if best["enclosure"] >= 0.75 else 0.1))

    chosen = max(analyses, key=score)
    for cand in sorted(analyses, key=lambda a: -float(a["plane"]["point"] @ z)):
        depth_below = float((chosen["plane"]["point"] - cand["plane"]["point"]) @ z)
        if cand is chosen or depth_below < 2 * FACE_TOL_M or cand["best"]["fill"] < 0.85 or score(cand) <= 0.02 * score(chosen):
            continue
        cg, cb = chosen["grid"], chosen["best"]
        centre = cand["grid"].origin + cand["best"]["target_uv"][0] * cand["grid"].u + cand["best"]["target_uv"][1] * cand["grid"].v
        rel = centre - cg.origin
        uv = np.array([rel @ cg.u, rel @ cg.v])
        i, j = np.floor((uv + cg.half) / cg.cell).astype(int)
        members = set(zip(*cb["cells"]))
        if (i, j) in members and cand["best"]["area_m2"] < 0.8 * cb["area_m2"]:
            chosen = cand
    grid, best = chosen["grid"], chosen["best"]
    n = grid.n
    mouth = grid.origin + best["target_uv"][0] * grid.u + best["target_uv"][1] * grid.v
    yaw, yaw_observed = _yaw_correction(best["angle"], best["aspect"])
    x_axis = _rotation_about(n, yaw) @ grid.u
    rotation = np.column_stack([x_axis, np.cross(n, x_axis), n])
    seated = np.eye(4)
    seated[:3, :3] = rotation
    floor = best.get("floor")
    face_target = mouth - n * seated_depth
    # The rim plane defines the mouth (seat = mouth - seated_depth) unless no rim encloses the
    # opening (deep family: lipped connectors) or the floor seen through the opening sits above
    # that seat (impossible: the rim found is a chamfer below the real mouth). Then the floor is the seat.
    floor_used = floor is not None and (best["family"] == "deep" or floor["height_m"] > -seated_depth + 0.0003)
    if floor_used:
        seated[:3, 3] = mouth + n * floor["height_m"]
    else:
        seated[:3, 3] = face_target

    # Multi-view consistency (jackknife): re-find the opening with each view left out, on a local grid.
    local = Grid(origin=grid.origin + best["centre_uv"][0] * grid.u + best["centre_uv"][1] * grid.v,
                 u=grid.u, v=grid.v, n=n, cell=grid.cell,
                 half=min(grid.half, float(np.abs(grid.uv(*best["cells"]) - best["centre_uv"]).max()) + 0.003))
    shift = best["centre_uv"]
    per_view = []
    for k in range(len(view_data)):
        subset = [data for m, data in enumerate(view_data) if m != k and len(data[0])]
        if not subset:
            continue
        holes = find_holes(classify(subset, local), local, best["family"])
        cands = [c for c in hypotheses(holes, local) if c["kind"] == best["kind"]]
        if best["kind"] == "opening" and best["aspect"] >= 5.0:  # slot: one view may see only part of its length
            ref = np.array([-math.sin(best["angle"]), math.cos(best["angle"])])
            cands = [c for c in cands if abs(c["centre_uv"] @ ref) < 0.001
                     and abs(_wrap(c["angle"] - best["angle"], math.pi)) < math.radians(5)]
        else:
            cands = [c for c in cands if np.linalg.norm(c["centre_uv"]) < 0.003]
        if cands:
            c = min(cands, key=lambda c: np.linalg.norm(c["centre_uv"]))
            per_view.append({"view": k, "centre_uv": c["centre_uv"] + shift, "angle": c["angle"], "aspect": c["aspect"],
                             "length_m": c["length_m"]})
    axis = np.array([math.cos(best["angle"]), math.sin(best["angle"])])
    normal_uv = np.array([-axis[1], axis[0]])
    across = [float((p["centre_uv"] - best["centre_uv"]) @ normal_uv) for p in per_view]
    along = [float((p["centre_uv"] - best["centre_uv"]) @ axis) for p in per_view]
    per_view_normals = []
    for points, _, _ in view_data:
        near = points[np.abs((points - grid.origin) @ n) < FACE_TOL_M] if len(points) else points
        if len(near) > 200:
            c = near.mean(axis=0)
            _, _, vt = np.linalg.svd(near - c, full_matrices=False)
            per_view_normals.append(vt[2] if vt[2] @ n > 0 else -vt[2])
    tilt_spread = float(max((math.acos(np.clip(m @ n, -1, 1)) for m in per_view_normals), default=0.0))
    yaw_spread = 0.0
    if yaw_observed:
        spreads = [abs(_wrap(p["angle"] - best["angle"], math.pi)) for p in per_view if p["aspect"] >= 1.4]
        yaw_spread = float(max(spreads, default=0.0))
    key = best.get("key")
    keyed_slot = best["kind"] == "opening" and best["aspect"] >= 5.0
    return {
        "seated_world": seated,
        "mouth_world": mouth,
        "plane_rms_m": chosen["plane"]["rms_m"],
        "plane_inliers": chosen["plane"]["inliers"],
        "n_planes": len(planes),
        "kind": best["kind"],
        "family": best["family"],
        "floor_used": floor_used,
        "floor_minus_face_m": float(floor["height_m"] + seated_depth) if floor is not None else 0.0,
        "floor_mad_m": float(floor["mad_m"]) if floor is not None else 0.0,
        "slot": keyed_slot,
        "key_found": key is not None,
        "key_candidates": key["candidates"] if key else 0,
        "enclosure": best["enclosure"],
        "hole_area_m2": best["area_m2"],
        "opening_length_m": best["length_m"],
        "aspect": best["aspect"],
        "yaw_observed": yaw_observed,
        "prior_offset_m": best["distance_m"],
        "prior_axial_offset_m": float((grid.origin - prior.mouth) @ n),
        "n_views_consistent": len(per_view),
        "across_spread_m": float(max(map(abs, across), default=0.0)),
        "along_spread_m": float(max(map(abs, along), default=0.0)),
        "tilt_spread_rad": tilt_spread,
        "yaw_spread_rad": yaw_spread,
        "_chosen": chosen,
        "_view_data": view_data,
    }



# --------------------------------------------------------------------------- uncertainty
def regime(diag: Mapping) -> str:
    """Which kind of measurement produced the estimate (decides its error floor)."""
    if diag.get("kind") == "pair":
        return "pin_pair"
    if diag.get("slot"):
        if not diag.get("key_found"):
            return "slot_no_key"
        return "slot_floor" if diag.get("floor_used") else "slot_key"
    return "opening"


#: Per-regime error floors (bound = floor + SPREAD_GAIN * measured spread, x2 when the views
#: disagree). Fitted against truth on calibration seeds 101-130 by perception/calibrate_uncertainty.py
#: (max residual x 1.5 per regime and component), never on the evaluation seeds; slot_no_key
#: never occurred there and keeps its hand-set 3 mm along-slot floor.
REGIME_FLOORS: dict[str, dict[str, float]] = {
    "opening": {"lateral_m": 0.00015, "axial_m": 0.00015, "tilt_rad": math.radians(0.10), "yaw_rad": math.radians(0.45)},
    "pin_pair": {"lateral_m": 0.00015, "axial_m": 0.00015, "tilt_rad": math.radians(0.10), "yaw_rad": math.radians(0.20)},
    "slot_floor": {"lateral_m": 0.00065, "axial_m": 0.00015, "tilt_rad": math.radians(0.39), "yaw_rad": math.radians(0.20)},
    "slot_key": {"lateral_m": 0.00047, "axial_m": 0.00015, "tilt_rad": math.radians(0.10), "yaw_rad": math.radians(0.20)},
    "slot_no_key": {"lateral_m": 0.00300, "axial_m": 0.00050, "tilt_rad": math.radians(1.00), "yaw_rad": math.radians(1.00)},
}
#: Largest face-plane residual seen on the calibration seeds (0.053 mm) plus 10 %. Recalibrate
#: (perception_eval.py --depth-noise ... then calibrate_uncertainty.py) for a real depth sensor.
CALIBRATED_PLANE_RMS_M = 0.00006
SPREAD_GAIN = {"lateral_m": 1.0, "axial_m": 1.0, "tilt_rad": 1.0, "yaw_rad": 1.0}


def spread_terms(diag: Mapping) -> dict[str, float]:
    """Measured disagreement per component: jackknife over views, plane residual, floor spread."""
    lateral = math.hypot(diag["across_spread_m"], 0.0 if diag["slot"] else diag["along_spread_m"])
    axial = 3.0 * diag["plane_rms_m"] + (diag.get("floor_mad_m", 0.0) if diag.get("floor_used") else 0.0)
    return {"lateral_m": lateral, "axial_m": axial, "tilt_rad": diag["tilt_spread_rad"],
            "yaw_rad": diag["yaw_spread_rad"] if diag["yaw_observed"] else 0.0}


def uncertainty_bounds(diag: Mapping, floors: Mapping = REGIME_FLOORS, gain: Mapping = SPREAD_GAIN) -> dict[str, float]:
    weak = diag["n_views_consistent"] < 2 or diag["enclosure"] < 0.8
    inflate = 2.0 if weak else 1.0
    floor = floors[regime(diag)]
    spread = spread_terms(diag)
    bounds = {k: inflate * (floor[k] + gain[k] * spread[k]) for k in floor}
    if not diag["yaw_observed"]:
        bounds["yaw_rad"] = PRIOR_BOUNDS["yaw_rad"]
    return bounds


# --------------------------------------------------------------------------- entry point
def _plain(value: Any) -> Any:
    """JSON-safe python scalars."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def prompts_for(task_name: str, language: str) -> list[str]:
    return SOCKET_PROMPTS.get(task_name, DEFAULT_PROMPTS)


def estimate_socket(adapter: Any, client: VisionClient | None = None, *, keep_views: bool = False,
                    noise: Mapping[str, float] | None = None) -> dict:
    """Vision estimate of the seated tip pose (world) with bounds; falls back to the prior."""
    started = time.time()
    env = adapter.native_env
    metadata = adapter.metadata
    seated_depth = float(env.seated_depth)
    prior = coarse_prior(adapter)
    prompts = prompts_for(metadata.task_name, metadata.language)
    client = client or VisionClient()
    rng = np.random.default_rng(int(getattr(adapter, "_seed", 0) or 0) + 7919)
    noise = dict(noise or {})
    timing: dict[str, float] = {}
    attempts = []
    result = None
    for variant in (0, 1):
        t0 = time.time()
        views = render_look_views(adapter, prior, variant)
        if noise:
            add_sensor_noise(views, np.random.default_rng(rng.integers(1 << 31)),
                             float(noise.get("depth_sigma_m", 0.0)), float(noise.get("extrinsic_m", 0.0)))
        timing[f"render_{variant}_s"] = time.time() - t0
        t0 = time.time()
        try:
            rois = [_prior_roi(v, prior, prior.window_m * 1.6).tolist() for v in views]
            detections = client.detect([v.rgb for v in views], prompts, rois)
        except Exception as exc:  # service down: geometry-only ROI from the prior
            detections = [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": [], "detector": [],
                           "masks": np.zeros((0,) + v.depth.shape, bool), "mask_scores": np.zeros(0)} for v in views]
            attempts.append({"variant": variant, "detector_error": f"{type(exc).__name__}: {exc}"})
        timing[f"detect_{variant}_s"] = time.time() - t0
        for view, det in zip(views, detections):
            view.detection = choose_detection(view, det, prior)
            if keep_views:
                view.detection["_raw"] = det
        detected = int(sum(v.detection["index"] is not None for v in views))
        det_score = float(np.mean([v.detection["score"] for v in views]))
        t0 = time.time()
        try:
            diag = estimate_from_views(views, prior, seated_depth, rng)
            diag.update(views_detected=detected, detection_score=det_score, variant=variant)
            # The error model holds only for depth as clean as on the calibration seeds: a face plane
            # noisier than that is out of calibration, and the prior is the honest answer.
            in_calibration = diag["plane_rms_m"] <= CALIBRATED_PLANE_RMS_M
            diag["in_calibration"] = in_calibration
            ok = diag["n_views_consistent"] >= 2 and (diag["key_found"] or not diag["slot"]) and in_calibration
            attempts.append({"variant": variant, "ok": bool(ok), "views_detected": detected,
                             "consistent_views": diag["n_views_consistent"]})
            if result is None or (ok and not result[1]):
                result = (diag, ok, views)
        except ValueError as exc:
            attempts.append({"variant": variant, "ok": False, "error": str(exc), "views_detected": detected})
        timing[f"geometry_{variant}_s"] = time.time() - t0
        if result is not None and result[1]:
            break
    record: dict[str, Any] = {"source": "vision", "prompts": prompts, "attempts": attempts}
    if result is None:
        seated = np.eye(4)
        seated[:3, :3] = prior.rotation
        seated[:3, 3] = prior.mouth - prior.rotation[:, 2] * seated_depth
        record.update(status="fallback_prior", seated_world=seated, bounds=dict(PRIOR_BOUNDS))
        views = views if keep_views else None
    else:
        diag, ok, views = result
        bounds = uncertainty_bounds(diag)
        seated = diag["seated_world"]
        if not ok:
            # Checks failed (views disagree, no detection, keyless slot): report the prior, whose
            # error is bounded, rather than a measurement that may be of the wrong feature.
            seated = np.eye(4)
            seated[:3, :3] = prior.rotation
            seated[:3, 3] = prior.mouth - prior.rotation[:, 2] * seated_depth
            bounds = dict(PRIOR_BOUNDS)
        status = "ok" if ok else "weak_prior"
        if ok and diag["views_detected"] == 0:
            # Geometry agrees across views but no detector confirmed the socket (service down,
            # or nothing gated): keep the measurement, double its bounds.
            status = "geometry_only"
            bounds = {k: 2.0 * v for k, v in bounds.items()}
        record.update(status=status, seated_world=seated, bounds=bounds,
                      diagnostics={k: _plain(v) for k, v in diag.items()
                                   if not k.startswith("_") and k not in ("seated_world", "mouth_world")})
        if keep_views:
            record["_diag"] = diag
    if os.environ.get("INSERTION_VISION_BOUNDS") == "spec":
        # Ablation only: keep the vision pose but report the synthetic estimate's spec-sheet
        # bounds, to separate the effect of the pose from the effect of the bounds on a program.
        from cap_harness.insertion import adapter as _adapter

        record["measured_bounds"] = record["bounds"]
        record["bounds"] = {"lateral_m": _adapter.SOCKET_LATERAL_M, "axial_m": _adapter.SOCKET_AXIAL_M,
                            "tilt_rad": _adapter.SOCKET_TILT_RAD, "yaw_rad": _adapter.SOCKET_YAW_RAD}
    timing["total_s"] = time.time() - started
    record["timing"] = timing
    record["prior_seated_world"] = np.vstack([np.column_stack([prior.rotation, prior.mouth - prior.rotation[:, 2] * seated_depth]),
                                              [0, 0, 0, 1]])
    if keep_views:
        record["_views"] = views
        record["_prior"] = prior
    return record
