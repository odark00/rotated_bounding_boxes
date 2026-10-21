"""
TASLER parcel dataset.

Reads sharded BlenderProc exports:

  TASLER_dir/shard_XXXXXXXX/{idx:08d}.gray.png   (grayscale render, used as model input)
  TASLER_dir/shard_XXXXXXXX/{idx:08d}.json       (scene / camera / label metadata)

(the .rgb/.depth/.seg/.seg_color siblings exist but are not consumed by this loader)

Each JSON describes exactly one labelled target parcel via:
  camera_intrinsics : K (3x3) / fx,fy,cx,cy / width,height  (pinhole; lens distortion ignored)
  camera_pose       : height_m, tilt_deg, roll_deg
  scene             : object_position [x,y], object_rotation_z (deg), object_world_aabb_m
  object_dimensions_m: width_m, length_m, height_m   ("fork_aligned_measured" local frame)

Camera model
------------
The renderer mounts the camera at world (0, 0, height_m) looking straight down the
world -Z axis, with a small extra pitch (tilt_deg, about the camera's right axis) and
roll (roll_deg, about the camera's forward/optical axis). This was reverse-engineered
and validated against the shipped depth maps and segmentation masks (projected 3D
boxes line up with the segmented parcel to within a few pixels across many samples;
projected center depth matches the rendered depth buffer up to the expected
surface-vs-center offset).

Concretely, in world coordinates (Z up):
  cam_pos  = (0, 0, height_m)
  right    = world +X, then rolled about `forward`
  down     = world -Y, then pitched about `right` (tilt), then rolled about `forward`
  forward  = world -Z, then pitched about `right` (tilt)

`(right, down, forward)` is already an OpenCV-style camera basis (x right, y down,
z forward-into-scene), so points expressed in it need no further axis flip before
projecting with K — unlike the previous Blender-exported dataset.

Produces per sample (N = 0 or 1 target parcel):
  - centers    (N,3)  3D center in camera frame (meters, OpenCV convention)
  - uv_norm    (N,2)  projected 2D center in [0,1] (model input resolution)
  - uv_pixel   (N,2)  projected 2D center in pixels (model input resolution)
  - depth      (N,1)  positive depth (meters, +Z forward)
  - sizes      (N,3)  metric (height, width, length) in meters — index 0 is vertical
  - rot6d      (N,6)  6D continuous rotation (object axes expressed in camera frame)
  - R          (N,3,3) rotation matrix (object axes as columns, camera frame)
  - boxes_3d   (N,8,3) 8 corners in camera frame, meters (bottom 0-3, top 4-7)
  - rboxes     (N,5)  2D oriented box (cx_px, cy_px, w_px, h_px, angle_rad) of the
                       face nearest the camera, in model-resolution pixels

Plus per-image:
  - K          (3,3)  intrinsics rescaled to model input resolution
  - K_original (3,3)  intrinsics matching the original render resolution
  - img_size   (2,)   model input size (W, H)
  - orig_size  (2,)   original render size (W, H) from JSON
  - num_parcels (int)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from geometry import matrix_to_rot6d
from model import TYPE_NAMES

# JSON `type` -> type id. Every TASLER_MULTI parcel type ("box", "prism",
# "padded_mailer", ...) is a loose parcel (id 0); TASLER_PACK's shipping units
# are id 1. See model.py for why the type is an explicit prediction.
TYPE_IDS = {name: i for i, name in enumerate(TYPE_NAMES)}


# ---------------------------------------------------------------------------
# Camera model (see module docstring)
# ---------------------------------------------------------------------------
def _rotate_about(vec: torch.Tensor, axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Rodrigues rotation of `vec` around unit `axis` by `angle` radians."""
    axis = axis / axis.norm()
    return (vec * math.cos(angle)
            + torch.linalg.cross(axis, vec) * math.sin(angle)
            + axis * (axis @ vec) * (1 - math.cos(angle)))


def camera_basis(tilt_deg: float, roll_deg: float) -> torch.Tensor:
    """
    Returns a (3,3) matrix R_cw whose rows are (right, down, forward) expressed in
    world coordinates. For a world vector v, R_cw @ v gives its camera-frame (x,y,z)
    components (OpenCV convention: x right, y down, z forward).
    """
    tilt = math.radians(tilt_deg)
    roll = math.radians(roll_deg)

    right0 = torch.tensor([1.0, 0.0, 0.0])
    down0 = torch.tensor([0.0, -1.0, 0.0])
    fwd0 = torch.tensor([0.0, 0.0, -1.0])

    fwd = _rotate_about(fwd0, right0, tilt)
    down = _rotate_about(down0, right0, tilt)
    right = _rotate_about(right0, fwd, roll)
    down = _rotate_about(down, fwd, roll)

    return torch.stack([right, down, fwd], dim=0)  # (3,3)


def world_to_camera(points_world: torch.Tensor, cam_pos: torch.Tensor,
                     R_cw: torch.Tensor) -> torch.Tensor:
    """points_world: (...,3) -> camera-frame (...,3), OpenCV convention."""
    rel = points_world - cam_pos
    return rel @ R_cw.T


def project(points_cam: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pinhole projection, OpenCV convention (x right, y down, +z forward). No axis flip.

    Args:
        points_cam: (N,3) camera-frame points
        K:          (3,3) intrinsics
    Returns:
        uv:    (N,2) pixel coords
        depth: (N,)  positive depth (meters)
    """
    z = torch.clamp(points_cam[:, 2:3], min=1e-6)
    xy = points_cam[:, :2] / z
    uv = xy @ K[:2, :2].T + K[:2, 2]
    return uv, points_cam[:, 2]


def world_box_corners(position_xy, rotation_z_deg: float,
                       width_m: float, length_m: float,
                       z0: float, z1: float) -> torch.Tensor:
    """
    Oriented box corners in world coordinates: bottom face 0-3, top face 4-7 (nearer
    the ceiling-mounted camera). Rotation is CCW about world +Z by `rotation_z_deg`,
    matching the JSON's `scene.object_rotation_z`.
    """
    hw, hl = width_m / 2.0, length_m / 2.0
    local = torch.tensor([
        [-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl],
    ], dtype=torch.float32)

    th = math.radians(rotation_z_deg)
    c, s = math.cos(th), math.sin(th)
    Rz = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    rot = local @ Rz.T  # (4,2)

    ox, oy = position_xy
    center_xy = torch.tensor([ox, oy], dtype=torch.float32)
    rot = rot + center_xy

    bottom = torch.cat([rot, torch.full((4, 1), z0)], dim=-1)
    top = torch.cat([rot, torch.full((4, 1), z1)], dim=-1)
    return torch.cat([bottom, top], dim=0)  # (8,3)


# ---------------------------------------------------------------------------
# Augmentation (grayscale-friendly, intrinsics-safe)
# ---------------------------------------------------------------------------
def rot90_target(target: Dict[str, Any], k: int, S: int) -> Dict[str, Any]:
    """
    Rotate a sample's labels to match `torch.rot90(img, k, dims=(-2, -1))`
    (k quarter-turns counter-clockwise) on a square SxS image.

    Only the four proper rotations of the square are used -- never a mirror.
    A reflection would map the object rotation matrix `R` to an improper matrix
    (det = -1), silently corrupting `rot6d`; a rotation keeps every label in this
    dict valid, including the 3D ones the current detector does not consume.

    Image-plane mapping for one CCW quarter-turn is (x, y) -> (y, S - x), which in
    the camera frame is the proper rotation (X, Y, Z) -> (Y, -X, Z) about the
    optical axis. Intrinsics follow: fx/fy and cx/cy swap, and the new cy is
    S - cx. (For TASLER these are all fx == fy and cx == cy == S/2, so K comes
    back unchanged, but the general form is applied so the augmentation stays
    correct if the intrinsics ever stop being centred.)
    """
    k %= 4
    if k == 0:
        return target

    out = dict(target)
    for _ in range(k):
        rb = out["rboxes"]
        if rb.numel():
            cx, cy = rb[:, 0].clone(), rb[:, 1].clone()
            rb = rb.clone()
            rb[:, 0] = cy
            rb[:, 1] = S - cx
            # long edge direction (cos t, sin t) -> (sin t, -cos t)  =>  t - pi/2
            rb[:, 4] = (rb[:, 4] - math.pi / 2) % math.pi
            out["rboxes"] = rb

        uv = out["uv_pixel"]
        if uv.numel():
            ux, uy = uv[:, 0].clone(), uv[:, 1].clone()
            uv = uv.clone()
            uv[:, 0] = uy
            uv[:, 1] = S - ux
            out["uv_pixel"] = uv
            out["uv_norm"] = uv / S

        # camera frame: (X, Y, Z) -> (Y, -X, Z)
        Rz = torch.tensor([[0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
        for key in ("centers", "boxes_3d"):
            if out[key].numel():
                out[key] = out[key] @ Rz.T
        if out["R"].numel():
            out["R"] = Rz @ out["R"]
            out["rot6d"] = torch.stack([matrix_to_rot6d(r) for r in out["R"]])

        K = out["K"].clone()
        fx, fy, kcx, kcy = K[0, 0].clone(), K[1, 1].clone(), K[0, 2].clone(), K[1, 2].clone()
        K[0, 0], K[1, 1] = fy, fx
        K[0, 2], K[1, 2] = kcy, S - kcx
        out["K"] = K

    return out


# Per-object keys that must be filtered together when augmentation drops an
# object that has left the frame.
_PER_OBJECT_TENSORS = ("centers", "uv_norm", "uv_pixel", "depths", "sizes",
                       "rot6d", "R", "boxes_3d", "rboxes", "labels", "obj_type")


def scale_target(img, target, s, S, min_edge_px=6.0):
    """
    Zoom a sample by `s` about a random anchor, keeping the SxS canvas: crop
    when s > 1, pad with black when s < 1 (black is also what the real
    packstuecke frames are padded with).

    This is an exact camera operation - a crop and a change of focal length -
    so only the 2D labels and K move; depth, metric size and the 3D corners are
    untouched. It is valid here only because depth is predicted in canonical
    units (depth_canon = depth * FX_REF / fx, Cube R-CNN's "virtual depth"):
    zooming by s divides depth_canon by s exactly as it multiplies the pixel
    extents, so the image and its labels stay consistent. With metric depth as
    the target, the same augmentation would teach the net that a bigger-looking
    object is at the same distance, which is false.

    What it buys: without it, a unit's pixel size and its depth are tied together
    by the renderer, and the net can read scale off absolute pixel size instead of
    off the object. The real packstuecke crop factor varies frame to frame, so
    that shortcut does not transfer.
    """
    new = max(8, int(round(S * s)))
    sx = new / S                                    # the scale actually realised
    img2 = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(new, new), mode="bilinear",
        align_corners=False).squeeze(0)
    if new >= S:
        x0 = int(torch.randint(0, new - S + 1, ()).item())
        y0 = int(torch.randint(0, new - S + 1, ()).item())
        img2 = img2[:, y0:y0 + S, x0:x0 + S]
        ox, oy = -x0, -y0
    else:
        canvas = torch.full((img.shape[0], S, S), -1.0, dtype=img.dtype)  # black
        x0 = int(torch.randint(0, S - new + 1, ()).item())
        y0 = int(torch.randint(0, S - new + 1, ()).item())
        canvas[:, y0:y0 + new, x0:x0 + new] = img2
        img2 = canvas
        ox, oy = x0, y0

    out = dict(target)
    K = out["K"].clone()
    K[0, 0] *= sx; K[1, 1] *= sx
    K[0, 2] = K[0, 2] * sx + ox
    K[1, 2] = K[1, 2] * sx + oy
    out["K"] = K

    rb = out["rboxes"].clone()
    if rb.numel():
        rb[:, 0] = rb[:, 0] * sx + ox
        rb[:, 1] = rb[:, 1] * sx + oy
        rb[:, 2:4] *= sx
        out["rboxes"] = rb
        uv = out["uv_pixel"].clone()
        uv[:, 0] = uv[:, 0] * sx + ox
        uv[:, 1] = uv[:, 1] * sx + oy
        out["uv_pixel"] = uv
        out["uv_norm"] = uv / S

        # drop what the crop pushed out of frame or shrank below detectability,
        # exactly as the dataset builders do - an unlabelled visible object is
        # a false negative the detector gets punished for
        keep = ((rb[:, 0] >= 0) & (rb[:, 0] < S) & (rb[:, 1] >= 0) & (rb[:, 1] < S)
                & (rb[:, 2] >= min_edge_px))
        if not bool(keep.all()):
            idx = keep.nonzero(as_tuple=True)[0]
            for key in _PER_OBJECT_TENSORS:
                if key in out and torch.is_tensor(out[key]) and out[key].shape[:1] == keep.shape:
                    out[key] = out[key][idx]
            out["names"] = [n for n, k in zip(out.get("names", []), keep.tolist()) if k]
            out["num_parcels"] = int(keep.sum())
    return img2.contiguous(), out


class PhotometricAug:
    """Photometric only — does not alter geometry, so K & 3D labels stay valid."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(()).item() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32)
        if torch.rand(()).item() < 0.7:
            alpha = 0.8 + 0.4 * torch.rand(()).item()   # contrast
            beta = -10 + 20 * torch.rand(()).item()     # brightness
            arr = np.clip(alpha * arr + beta, 0, 255)
        if torch.rand(()).item() < 0.4:
            sigma = 2 + 6 * torch.rand(()).item()
            arr = np.clip(arr + np.random.randn(*arr.shape) * sigma, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ParcelDataset(Dataset):
    """
    TASLER shard dataset. Geometry-aware targets built from the world-frame scene
    description (see module docstring for the camera model).

    Args:
        data_dir:   TASLER root folder (or list of roots, pooled together)
                    containing shard_XXXXXXXX/ subfolders. Passing multiple
                    roots (e.g. TASLER_MULTI + TASLER) lets differently
                    labelled datasets be trained on jointly, since each root
                    is scanned independently and shard names never need to
                    be unique across roots.
        img_size:   model input size (square). K is rescaled accordingly.
        max_parcels: cap on labelled objects per image (multi-object schema
                     only; each single-object TASLER sample has at most 1).
        augment:    whether to apply photometric augmentation (contrast /
                    brightness / noise; geometry untouched)
        scale_augment: random zoom in `scale_range` (crop or black-pad back to
                    the canvas) with K updated to match - see `scale_target`.
                    Training only.
        geom_augment: whether to also apply a random quarter-turn rotation.
                    Labels are rotated with the image (see `rot90_target`), so
                    this is a 4x exact enlargement of a dataset that is
                    otherwise only ~3.7k images. Enable for training only.
        normalize:  if True, normalize to mean=0.5/std=0.5
    """

    def __init__(
        self,
        data_dir: str | Path | List[str | Path],
        img_size: int = 640,
        max_parcels: int = 10,
        augment: bool = False,
        geom_augment: bool = False,
        scale_augment: bool = False,
        scale_range: Tuple[float, float] = (0.5, 1.25),
        scale_prob: float = 0.7,
        normalize: bool = True,
        verbose: bool = False,
    ):
        roots = [data_dir] if isinstance(data_dir, (str, Path)) else list(data_dir)
        self.data_dirs = [Path(r) for r in roots]
        self.img_size = img_size
        self.max_parcels = max_parcels
        self.augment = augment
        self.geom_augment = geom_augment
        self.scale_augment = scale_augment
        self.scale_range = scale_range
        self.scale_prob = scale_prob
        self.verbose = verbose

        self.samples: List[Tuple[Path, str]] = []
        for root in self.data_dirs:
            for shard_dir in sorted(root.glob("shard_*")):
                if not shard_dir.is_dir():
                    continue
                for json_path in sorted(shard_dir.glob("*.json")):
                    stem = json_path.stem
                    if (shard_dir / f"{stem}.gray.png").exists():
                        self.samples.append((shard_dir, stem))

        if not self.samples:
            raise FileNotFoundError(
                f"No TASLER samples found under {self.data_dirs} "
                f"(expected shard_XXXXXXXX/*.gray.png + *.json)"
            )

        self.photo_aug = PhotometricAug(p=0.5) if augment else None
        tfm = [T.Resize((img_size, img_size)), T.ToTensor()]
        if normalize:
            tfm.append(T.Normalize(mean=[0.5], std=[0.5]))
        self.transform = T.Compose(tfm)

        if self.verbose:
            print(f"[ParcelDataset] {len(self.samples)} samples from {self.data_dirs}")

    # ------------------------------------------------------------------ utils
    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _build_K(label: Dict[str, Any]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        ci = label["camera_intrinsics"]
        K = torch.tensor([
            [ci["fx"], 0.0, ci["cx"]],
            [0.0, ci["fy"], ci["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        return K, (int(ci["width"]), int(ci["height"]))

    @staticmethod
    def _rescale_K(K: torch.Tensor, src_wh: Tuple[int, int],
                   dst_wh: Tuple[int, int]) -> torch.Tensor:
        """Rescale intrinsics when image is resized (no cropping)."""
        sx = dst_wh[0] / src_wh[0]
        sy = dst_wh[1] / src_wh[1]
        K2 = K.clone()
        K2[0] *= sx
        K2[1] *= sy
        return K2

    @staticmethod
    def _rbox_from_top_face(top4_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        2D oriented bbox (cx, cy, w, h, angle) from the 4 corners of the box face
        nearest the camera, in model-resolution pixels.

        **Long-edge convention**: `w >= h` always, and `angle` (in [0, pi)) is the
        image-plane direction of the *long* edge.

        The previous convention tied `w`/`angle` to the parcel's own "width" axis,
        which makes the label unlearnable: the same pixels are described equally
        well by (w, h, angle) and (h, w, angle + pi/2), and which one the label
        carries depends on the parcel's internal axis naming, something no image
        can reveal. 41% of TASLER parcels project to a near-square top face
        (long/short < 1.15), so the old target flipped by 90 degrees essentially
        at random on half the dataset -- the direct cause of the ~41 deg angle MAE
        the earlier runs reported. Sorting the edges by length makes the target a
        function of the image alone.

        For a genuinely square top face the long edge is still arbitrary, but that
        ambiguity is now intrinsic to the shape rather than to the labelling, and
        losses.py down-weights the angle term by box elongation accordingly.
        """
        uv, _ = project(top4_cam, K)  # (4,2)
        cx = uv[:, 0].mean()
        cy = uv[:, 1].mean()

        e1 = uv[1] - uv[0]  # parcel width-edge direction in 2D
        e2 = uv[3] - uv[0]  # parcel length-edge direction in 2D

        w = e1.norm().clamp(min=1e-3)
        h = e2.norm().clamp(min=1e-3)

        if h > w:            # keep the long edge in `w`, and point `angle` along it
            w, h = h, w
            e1 = e2

        angle = torch.atan2(e1[1], e1[0]) % torch.pi

        return torch.stack([cx, cy, w, h, angle])

    # ------------------------------------------------------------------ main
    def __getitem__(self, idx: int):
        shard_dir, stem = self.samples[idx]
        with open(shard_dir / f"{stem}.json", "r") as f:
            label = json.load(f)

        # ---- image ----
        img_path = shard_dir / f"{stem}.gray.png"
        img = Image.open(img_path).convert("L")
        orig_size = img.size  # (W, H)

        if self.photo_aug is not None:
            img = self.photo_aug(img)
        img_tensor = self.transform(img)  # (1, S, S)

        # ---- intrinsics (rescaled to model input resolution) ----
        K_orig, json_wh = self._build_K(label)
        K_model = self._rescale_K(K_orig, json_wh, (self.img_size, self.img_size))

        if "objects" in label:
            target = self._build_multi_target(label, K_model)
        else:
            target = self._build_single_target(label, K_model)

        target["has_3d"] = True
        target["stem"] = f"{shard_dir.name}/{stem}"
        target["image_path"] = str(img_path)
        target["K"] = K_model
        target["K_original"] = K_orig
        target["img_size"] = torch.tensor([self.img_size, self.img_size])
        target["orig_size"] = torch.tensor([orig_size[0], orig_size[1]])

        if self.scale_augment and torch.rand(()).item() < self.scale_prob:
            lo, hi = self.scale_range
            # log-uniform, so zooming out 2x is as likely as zooming in 2x
            s = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(()).item())
            img_tensor, target = scale_target(img_tensor, target, s, self.img_size)

        if self.geom_augment:
            k = int(torch.randint(0, 4, ()).item())
            if k:
                img_tensor = torch.rot90(img_tensor, k, dims=(-2, -1)).contiguous()
                target = rot90_target(target, k, self.img_size)

        return img_tensor, target

    # ------------------------------------------------------------------
    def _build_single_target(self, label: Dict[str, Any], K_model: torch.Tensor):
        """Original TASLER schema: exactly one labelled parcel, described via
        world-frame scene geometry + camera pose (reconstructed here)."""
        cam_pose = label["camera_pose"]
        cam_h = cam_pose["height_m"]
        R_cw = camera_basis(cam_pose["tilt_deg"], cam_pose["roll_deg"])  # (3,3)
        cam_pos = torch.tensor([0.0, 0.0, cam_h])

        scene = label["scene"]
        dims = label["object_dimensions_m"]
        aabb = scene["object_world_aabb_m"]
        z0, z1 = float(aabb[0][2]), float(aabb[1][2])

        corners_w = world_box_corners(
            scene["object_position"], scene["object_rotation_z"],
            dims["width_m"], dims["length_m"], z0, z1,
        )  # (8,3) world
        corners_c = world_to_camera(corners_w, cam_pos, R_cw)  # (8,3) camera frame, meters

        center_c = corners_c.mean(dim=0)  # (3,)
        uv_px, depth = project(center_c.unsqueeze(0), K_model)
        uv_px = uv_px.squeeze(0)
        depth = depth.squeeze(0)

        size = torch.tensor([dims["height_m"], dims["width_m"], dims["length_m"]])
        rbox = self._rbox_from_top_face(corners_c[4:8], K_model)  # (5,)

        th = math.radians(scene["object_rotation_z"])
        c, s = math.cos(th), math.sin(th)
        R_wo = torch.tensor([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        R = R_cw @ R_wo  # (3,3), columns = object axes in camera frame
        rot6d = matrix_to_rot6d(R)

        name = label.get("composition", {}).get("package_identity",
                                                  label.get("model_name", ""))
        n = 1

        return {
            "names": [name],
            "centers": center_c.unsqueeze(0),          # (1,3) camera frame, meters
            "uv_norm": (uv_px / self.img_size).unsqueeze(0),  # (1,2) in [0,1]
            "uv_pixel": uv_px.unsqueeze(0),             # (1,2) pixels, model res
            "depths": depth.unsqueeze(0).unsqueeze(0),  # (1,1) meters
            "sizes": size.unsqueeze(0),                 # (1,3) meters (h, w, l)
            "rot6d": rot6d.unsqueeze(0),                # (1,6)
            "R": R.unsqueeze(0),                        # (1,3,3)
            "boxes_3d": corners_c.unsqueeze(0),         # (1,8,3) camera frame, meters
            "rboxes": rbox.unsqueeze(0),                # (1,5) cx,cy,w,h,angle (model px)
            "labels": torch.ones(n, dtype=torch.long),
            "obj_type": torch.zeros(n, dtype=torch.long),   # parcel
            "num_parcels": n,
        }

    # ------------------------------------------------------------------
    def _build_multi_target(self, label: Dict[str, Any], K_model: torch.Tensor):
        """TASLER_MULTI schema: `objects[]`, each already carrying precomputed
        camera-frame corners (OpenCV convention, bottom0-3/top4-7 layout) — no
        camera-pose reconstruction needed, just rescale for the resized image
        and re-project with the resolution-matched K."""
        objs = label["objects"][: self.max_parcels]
        n = len(objs)

        names, centers, depths, sizes, rboxes, boxes_3d, rot6ds, Rs, types = (
            [], [], [], [], [], [], [], [], []
        )
        for o in objs:
            corners_c = torch.tensor(o["boxes_3d_cam"], dtype=torch.float32)  # (8,3) meters
            center_c = corners_c.mean(dim=0)
            uv_px, depth = project(center_c.unsqueeze(0), K_model)
            uv_px = uv_px.squeeze(0)
            depth = depth.squeeze(0)

            dims = o["object_dimensions_m"]
            size = torch.tensor([dims["height_m"], dims["width_m"], dims["length_m"]])
            rbox = self._rbox_from_top_face(corners_c[4:8], K_model)  # (5,)

            # object axes (width, length, up) directly from the box edges —
            # already expressed in camera frame, no world reconstruction needed.
            top4 = corners_c[4:8]
            u_w = (top4[1] - top4[0])
            u_w = u_w / u_w.norm().clamp(min=1e-6)
            u_l = (top4[3] - top4[0])
            u_l = u_l / u_l.norm().clamp(min=1e-6)
            u_h = torch.linalg.cross(u_w, u_l)
            u_h = u_h / u_h.norm().clamp(min=1e-6)
            R = torch.stack([u_w, u_l, u_h], dim=1)  # (3,3), columns = object axes
            rot6d = matrix_to_rot6d(R)

            names.append(o.get("name", ""))
            types.append(TYPE_IDS.get(o.get("type", ""), 0))
            centers.append(center_c)
            depths.append(depth.unsqueeze(0))
            sizes.append(size)
            rboxes.append(rbox)
            boxes_3d.append(corners_c)
            rot6ds.append(rot6d)
            Rs.append(R)

        if n == 0:
            return {
                "names": [],
                "centers": torch.zeros(0, 3),
                "uv_norm": torch.zeros(0, 2),
                "uv_pixel": torch.zeros(0, 2),
                "depths": torch.zeros(0, 1),
                "sizes": torch.zeros(0, 3),
                "rot6d": torch.zeros(0, 6),
                "R": torch.zeros(0, 3, 3),
                "boxes_3d": torch.zeros(0, 8, 3),
                "rboxes": torch.zeros(0, 5),
                "labels": torch.zeros(0, dtype=torch.long),
                "obj_type": torch.zeros(0, dtype=torch.long),
                "num_parcels": 0,
            }

        centers = torch.stack(centers)              # (n,3)
        depths = torch.cat(depths).unsqueeze(-1)     # (n,1)
        sizes = torch.stack(sizes)                   # (n,3)
        rboxes = torch.stack(rboxes)                 # (n,5)
        boxes_3d = torch.stack(boxes_3d)             # (n,8,3)
        rot6d = torch.stack(rot6ds)                  # (n,6)
        R = torch.stack(Rs)                          # (n,3,3)
        uv_px = rboxes[:, :2]                         # (n,2) pixels, model res

        return {
            "names": names,
            "centers": centers,
            "uv_norm": uv_px / self.img_size,
            "uv_pixel": uv_px,
            "depths": depths,
            "sizes": sizes,
            "rot6d": rot6d,
            "R": R,
            "boxes_3d": boxes_3d,
            "rboxes": rboxes,
            "labels": torch.ones(n, dtype=torch.long),
            "obj_type": torch.tensor(types, dtype=torch.long),
            "num_parcels": n,
        }


# ---------------------------------------------------------------------------
# Real frames with 2D labels only
# ---------------------------------------------------------------------------
class RealPackDataset(Dataset):
    """
    The real `packstuecke/` frames, for PARTIAL supervision.

    Each frame carries human-drawn rotated boxes (`*.txt`: `class cx cy w h
    angle_deg`, normalised) and nothing else: no calibration, no depth, no metric
    size. What those labels do pin down is exactly what the synthetic data could
    not teach transferably - where a real shipping unit is, its 2D outline, and
    that it IS a shipping unit (every object annotated here is one). So targets
    from this dataset set `has_3d=False`, and losses.py trains detection, box,
    angle and type on them while switching the metric-size and depth terms off.

    Framing matches `eval_packstuecke_real.py`: the frame goes top-left onto a
    square black canvas (what the 1280x1280 variants already are), then is
    resized to the model's input. Isotropic, so rotated boxes stay rotated boxes.

    The 3D fields are filled with inert placeholders of the right shapes, so
    `rot90_target` / `scale_target` / collate treat these samples exactly like
    synthetic ones. `K` is a placeholder too: nothing reads it for has_3d=False
    targets except the augmentations, which keep it self-consistent.
    """

    def __init__(self, data_dir, img_size=640, augment=False, geom_augment=False,
                 scale_augment=False, scale_range=(0.5, 1.25), scale_prob=0.7,
                 normalize=True, verbose=False):
        self.dir = Path(data_dir)
        self.img_size = img_size
        self.augment, self.geom_augment = augment, geom_augment
        self.scale_augment, self.scale_range, self.scale_prob = (
            scale_augment, scale_range, scale_prob)
        self.samples = []
        for txt in sorted(self.dir.glob("*.txt")):
            jpg = txt.with_suffix(".jpg")
            if txt.stem != "classes" and jpg.exists():
                self.samples.append((jpg, txt))
        if not self.samples:
            raise FileNotFoundError(f"no labelled *.jpg + *.txt pairs in {self.dir}")
        self.photo_aug = PhotometricAug(p=0.5) if augment else None
        tfm = [T.Resize((img_size, img_size)), T.ToTensor()]
        if normalize:
            tfm.append(T.Normalize(mean=[0.5], std=[0.5]))
        self.transform = T.Compose(tfm)
        if verbose:
            print(f"[RealPackDataset] {len(self.samples)} labelled frames from {self.dir}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _long_edge(cx, cy, w, h, ang):
        """Put a (w, h, angle) box into the long-edge convention the heads use."""
        if h > w:
            w, h = h, w
            ang = ang + math.pi / 2
        return [cx, cy, w, h, ang % math.pi]

    def __getitem__(self, idx):
        jpg, txt = self.samples[idx]
        img = Image.open(jpg).convert("L")
        W0, H0 = img.size
        side = max(W0, H0)
        if W0 != H0:
            canvas = Image.new("L", (side, side), 0)
            canvas.paste(img, (0, 0))
            img = canvas
        s = self.img_size / side

        boxes = []
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) != 6:
                continue
            try:
                _, cx, cy, w, h, ang = (float(v) for v in parts)
            except ValueError:
                continue
            boxes.append(self._long_edge(cx * W0 * s, cy * H0 * s, w * W0 * s,
                                         h * H0 * s, math.radians(ang)))

        if self.photo_aug is not None:
            img = self.photo_aug(img)
        img_tensor = self.transform(img)

        n = len(boxes)
        S = self.img_size
        rb = torch.tensor(boxes, dtype=torch.float32).reshape(n, 5)
        eye = torch.eye(3).expand(n, 3, 3).clone()
        K = torch.tensor([[float(S), 0.0, S / 2.0],
                          [0.0, float(S), S / 2.0],
                          [0.0, 0.0, 1.0]])
        target = {
            "names": [""] * n,
            "centers": torch.zeros(n, 3),
            "uv_pixel": rb[:, :2].clone(),
            "uv_norm": rb[:, :2] / S,
            "depths": torch.ones(n, 1),
            "sizes": torch.ones(n, 3),
            "rot6d": torch.stack([matrix_to_rot6d(r) for r in eye]) if n else torch.zeros(0, 6),
            "R": eye,
            "boxes_3d": torch.zeros(n, 8, 3),
            "rboxes": rb,
            "labels": torch.ones(n, dtype=torch.long),
            "obj_type": torch.full((n,), TYPE_IDS["palletised_load"], dtype=torch.long),
            "num_parcels": n,
            "has_3d": False,
            "stem": f"real/{jpg.stem}",
            "image_path": str(jpg),
            "K": K,
            "K_original": K.clone(),
            "img_size": torch.tensor([S, S]),
            "orig_size": torch.tensor([W0, H0]),
        }

        if self.scale_augment and torch.rand(()).item() < self.scale_prob:
            lo, hi = self.scale_range
            sc = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(()).item())
            img_tensor, target = scale_target(img_tensor, target, sc, S)
        if self.geom_augment:
            k = int(torch.randint(0, 4, ()).item())
            if k:
                img_tensor = torch.rot90(img_tensor, k, dims=(-2, -1)).contiguous()
                target = rot90_target(target, k, S)
        return img_tensor, target


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    """Stack images; keep targets as a list (variable N per sample)."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
