"""
Parcel 3D dataset.

Reads:
  images_dir/img_XXXX.png   (grayscale)
  labels_dir/img_XXXX.json  (Blender-exported labels)

Produces per parcel:
  - center_cam (3,)         3D center in Blender camera frame
  - uv_norm    (2,)         projected 2D center in [0,1]
  - uv_pixel   (2,)         projected 2D center in pixels (original image)
  - depth      (1,)         positive depth (OpenCV +Z forward)
  - size       (3,)         metric (w, h, d) in meters
  - rot6d      (6,)         6D continuous rotation
  - R          (3,3)        rotation matrix
  - euler      (3,)         original Blender XYZ euler (kept for reference)
  - corners_8  (8,3)        ground-truth 8 corners (Blender frame)

Plus per-image:
  - K          (3,3)        intrinsics (matches image resolution stored in JSON)
  - img_size   (2,)         original (W, H) from JSON
  - num_parcels (int)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from geometry import euler_zyx_to_matrix, matrix_to_rot6d


# ---------------------------------------------------------------------------
# Projection (Blender -> pixel)
# ---------------------------------------------------------------------------
def project_blender(points_cam: torch.Tensor, K: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project Blender-frame camera points to image pixels.

    Blender camera convention: x right, y up, -z forward.
    Convert to OpenCV (x right, y down, +z forward) before projecting.

    Args:
        points_cam: (N, 3) in Blender frame
        K:          (3, 3) intrinsics matching the rendered image resolution

    Returns:
        uv:    (N, 2) pixel coordinates
        depth: (N,)   positive depth (meters)
    """
    pts = points_cam.clone()
    pts[:, 1] = -pts[:, 1]
    pts[:, 2] = -pts[:, 2]
    z = torch.clamp(pts[:, 2:3], min=1e-6)
    xy = pts[:, :2] / z
    uv = xy @ K[:2, :2].T + K[:2, 2]
    return uv, pts[:, 2]


# ---------------------------------------------------------------------------
# Augmentation (grayscale-friendly, intrinsics-safe)
# ---------------------------------------------------------------------------
class PhotometricAug:
    """Photometric only — does not alter geometry, so K & 3D labels stay valid."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(()).item() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32)
        # random brightness/contrast
        if torch.rand(()).item() < 0.7:
            alpha = 0.8 + 0.4 * torch.rand(()).item()   # contrast
            beta  = -10 + 20 * torch.rand(()).item()    # brightness
            arr = np.clip(alpha * arr + beta, 0, 255)
        # gaussian noise
        if torch.rand(()).item() < 0.4:
            sigma = 2 + 6 * torch.rand(()).item()
            arr = np.clip(arr + np.random.randn(*arr.shape) * sigma, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ParcelDataset(Dataset):
    """
    Pairs of (grayscale image, JSON labels).  Geometry-aware targets.

    Args:
        images_dir: folder with img_XXXX.png
        labels_dir: folder with img_XXXX.json
        img_size:   model input size (square).  K is rescaled accordingly.
        max_parcels: maximum parcels per image; extras are dropped (warning printed).
        augment:    whether to apply photometric augmentation
        normalize:  if True, normalize to mean=0.5/std=0.5
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        img_size: int = 640,
        max_parcels: int = 10,
        augment: bool = False,
        normalize: bool = True,
        verbose: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.max_parcels = max_parcels
        self.augment = augment
        self.verbose = verbose

        # discover samples that have both image & json
        img_stems = {p.stem for p in self.images_dir.glob("*.png")}
        lbl_stems = {p.stem for p in self.labels_dir.glob("*.json")}
        self.stems = sorted(img_stems & lbl_stems)
        if not self.stems:
            raise FileNotFoundError(
                f"No matched pairs found.\n  images_dir={self.images_dir}\n  labels_dir={self.labels_dir}"
            )

        # image transforms
        self.photo_aug = PhotometricAug(p=0.5) if augment else None
        tfm = [T.Resize((img_size, img_size)), T.ToTensor()]
        if normalize:
            tfm.append(T.Normalize(mean=[0.5], std=[0.5]))
        self.transform = T.Compose(tfm)

        if self.verbose:
            print(f"[ParcelDataset] {len(self.stems)} samples "
                  f"from {self.images_dir.name}")

    # ------------------------------------------------------------------ utils
    def __len__(self) -> int:
        return len(self.stems)

    def _load_label(self, stem: str) -> Dict[str, Any]:
        with open(self.labels_dir / f"{stem}.json", "r") as f:
            return json.load(f)

    @staticmethod
    def _build_K(label: Dict[str, Any]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        ci = label["camera_intrinsics"]
        K = torch.tensor([
            [ci["fx"], 0.0,      ci["cx"]],
            [0.0,      ci["fy"], ci["cy"]],
            [0.0,      0.0,      1.0],
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
    def _size_from_corners(bbox: torch.Tensor) -> torch.Tensor:
        """
        Recover (w, h, d) from the 8 GT corners.

        Corner ordering from Blender export:
            0,1,2,3 — one face;  4,5,6,7 — opposite face
            edges (0,1), (1,2), (0,4) are three orthogonal cube edges.
        """
        d01 = (bbox[0] - bbox[1]).norm()
        d12 = (bbox[1] - bbox[2]).norm()
        d04 = (bbox[0] - bbox[4]).norm()
        return torch.stack([d01, d12, d04])

    # ------------------------------------------------------------------ main
    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        label = self._load_label(stem)

        # ---- image ----
        img_path = self.images_dir / f"{stem}.png"
        img = Image.open(img_path).convert("L")
        orig_size = img.size  # (W, H)

        if self.photo_aug is not None:
            img = self.photo_aug(img)
        img_tensor = self.transform(img)  # (1, S, S)

        # ---- intrinsics (rescaled to model input resolution) ----
        K_orig, json_wh = self._build_K(label)
        # use JSON's declared (W,H) as the source frame; resize target is (img_size, img_size)
        K_model = self._rescale_K(K_orig, json_wh, (self.img_size, self.img_size))

        # ---- parcels ----
        parcels = label.get("parcels", [])
        if len(parcels) > self.max_parcels:
            if self.verbose:
                print(f"[ParcelDataset] {stem}: truncating "
                      f"{len(parcels)} -> {self.max_parcels} parcels")
            parcels = parcels[: self.max_parcels]

        centers_list, uv_norm_list, uv_pixel_list = [], [], []
        depths_list, sizes_list = [], []
        rot6d_list, R_list, euler_list = [], [], []
        corners_list, names = [], []

        for p in parcels:
            bbox_b = torch.tensor(p["bbox_3d_camera"], dtype=torch.float32)  # (8,3)
            corners_list.append(bbox_b)

            # center in Blender frame
            center_b = bbox_b.mean(dim=0)                                     # (3,)
            centers_list.append(center_b)

            # project center using *model-resolution* K
            uv_px, depth = project_blender(center_b.unsqueeze(0), K_model)
            uv_px = uv_px.squeeze(0)        # (2,) pixels in model frame
            depth = depth.squeeze(0)        # scalar (positive)

            uv_pixel_list.append(uv_px)
            uv_norm_list.append(uv_px / self.img_size)
            depths_list.append(depth.unsqueeze(0))

            # metric size
            sizes_list.append(self._size_from_corners(bbox_b))

            # rotation
            euler = torch.tensor(p["rotation"], dtype=torch.float32)
            R = euler_zyx_to_matrix(euler)
            rot6d = matrix_to_rot6d(R)
            euler_list.append(euler)
            R_list.append(R)
            rot6d_list.append(rot6d)

            names.append(p.get("name", ""))

        n = len(parcels)

        def _stack(xs, shape):
            return torch.stack(xs, dim=0) if n else torch.zeros((0, *shape))

        target = {
            "stem":        stem,
            "image_path":  str(img_path),
            "names":       names,

            # geometry targets (model space, after resize)
            "centers":     _stack(centers_list, (3,)),       # (N,3) Blender frame
            "uv_norm":     _stack(uv_norm_list, (2,)),       # (N,2) in [0,1]
            "uv_pixel":    _stack(uv_pixel_list, (2,)),      # (N,2) pixels in img_size frame
            "depths":      _stack(depths_list,  (1,)),       # (N,1) meters, +Z
            "sizes":       _stack(sizes_list,   (3,)),       # (N,3) meters
            "rot6d":       _stack(rot6d_list,   (6,)),       # (N,6)
            "R":           _stack(R_list,       (3, 3)),     # (N,3,3)
            "euler":       _stack(euler_list,   (3,)),       # (N,3) original Blender XYZ
            "boxes_3d":    _stack(corners_list, (8, 3)),     # (N,8,3) Blender frame GT corners

            # bookkeeping
            "labels":      torch.ones(n, dtype=torch.long),  # 1 = parcel
            "K":           K_model,                          # (3,3) matches model resolution
            "K_original":  K_orig,                           # (3,3) matches original render
            "img_size":    torch.tensor([self.img_size, self.img_size]),
            "orig_size":   torch.tensor([orig_size[0], orig_size[1]]),
            "num_parcels": n,
        }

        return img_tensor, target


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    """Stack images; keep targets as a list (variable N per sample)."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets