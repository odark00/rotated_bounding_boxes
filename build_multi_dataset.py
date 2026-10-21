"""
Convert the synthetic_dataset_blender render output (multi-parcel scenes,
Blender camera convention) into a TASLER-style sharded dataset with MULTIPLE
labelled objects per image (OpenCV camera convention), consumable by
dataset.py's ParcelDataset.

Source (per image):
  synthetic_dataset_blender/gray_images/img_XXXX.png
  synthetic_dataset_blender/labels/img_XXXX.json
    - camera_intrinsics: fx, fy, cx, cy, width, height (ideal pinhole, RES x RES)
    - parcels[]: name, type, dimensions [dx,dy,dz] (local, unrotated: width,
      length, height), bbox_3d_camera: 8 corners in Blender camera space
      (x right, y up, z back — camera looks down -Z), in Blender's fixed
      bound_box corner order:
        0:(-x,-y,-z) 1:(-x,-y,+z) 2:(-x,+y,+z) 3:(-x,+y,-z)
        4:(+x,-y,-z) 5:(+x,-y,+z) 6:(+x,+y,+z) 7:(+x,+y,-z)
      (verified empirically against a unit cube via `blender -b -P`).

Output (per image):
  TASLER_MULTI/shard_XXXXXXXX/{idx:08d}.gray.png   (symlink to source PNG)
  TASLER_MULTI/shard_XXXXXXXX/{idx:08d}.json
    - camera_intrinsics: {fx, fy, cx, cy, width, height, source_camera_model}
    - objects[]: {name, type, object_dimensions_m: {width_m, length_m, height_m},
                  boxes_3d_cam: (8,3) list, OpenCV convention (x right, y down,
                  z forward), corner order bottom0-3/top4-7, cyclic
                  (-,-),(+,-),(+,+),(-,+) in local (width,length) — matching
                  dataset.py's world_box_corners() layout exactly, so no
                  camera-pose reconstruction is needed downstream: these are
                  already final camera-frame corners.}

Filtering (per object): keep only objects that are plausibly detectable —
in-frame center, minimum projected size, sane metric dimensions. Capped to
the MAX_OBJECTS_PER_IMAGE largest (by projected area) after filtering.
"""

import json
import shutil
from pathlib import Path

import numpy as np

SRC_ROOT = Path("/home/daryna/code_ws/study/3D_detection/synthetic_dataset_blender")
SRC_IMAGES = SRC_ROOT / "gray_images"
SRC_LABELS = SRC_ROOT / "labels"

OUT_ROOT = Path(__file__).resolve().parent / "TASLER_MULTI"
SHARD_SIZE = 250
MAX_OBJECTS_PER_IMAGE = 30

# Blender bound_box index groups (see module docstring)
BOTTOM_IDX = [0, 4, 7, 3]   # (-,-), (+,-), (+,+), (-,+) at local -z
TOP_IDX    = [1, 5, 6, 2]   # (-,-), (+,-), (+,+), (-,+) at local +z

MIN_DEPTH_M, MAX_DEPTH_M = 0.3, 15.0
MIN_RBOX_AREA_PX = 16.0
DIM_RANGES = dict(
    width_m=(0.02, 3.0),
    length_m=(0.02, 3.0),
    height_m=(0.03, 2.6),
)


def blender_cam_to_opencv(corners_blender):
    """(N,3) Blender camera frame (x right, y up, z back) -> OpenCV (x right,
    y down, z forward). Two sign flips = pure rotation, handedness preserved."""
    c = np.asarray(corners_blender, dtype=np.float64)
    return np.stack([c[:, 0], -c[:, 1], -c[:, 2]], axis=1)


def rbox_from_top_face(top4_cv, fx, fy, cx, cy):
    """2D oriented box (cx_px, cy_px, w_px, h_px, angle_rad) from 4 camera-frame
    corners of the nearest face, in cyclic (-,-),(+,-),(+,+),(-,+) order."""
    z = np.clip(top4_cv[:, 2], 1e-6, None)
    u = top4_cv[:, 0] / z * fx + cx
    v = top4_cv[:, 1] / z * fy + cy
    uv = np.stack([u, v], axis=1)

    center = uv.mean(axis=0)
    e1 = uv[1] - uv[0]   # width-edge
    e2 = uv[3] - uv[0]   # length-edge
    w = float(np.linalg.norm(e1))
    h = float(np.linalg.norm(e2))
    angle = float(np.arctan2(e1[1], e1[0]))
    if angle < 0:
        angle += np.pi
    return center, w, h, angle


def convert_one(label_path):
    label = json.loads(label_path.read_text())
    ci = label["camera_intrinsics"]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
    width, height = ci["width"], ci["height"]

    objects = []
    for p in label.get("parcels", []):
        dims = p.get("dimensions")
        corners_b = p.get("bbox_3d_camera")
        if not dims or not corners_b or len(corners_b) != 8:
            continue

        width_m, length_m, height_m = dims[0], dims[1], dims[2]
        ok = True
        for key, val in (("width_m", width_m), ("length_m", length_m), ("height_m", height_m)):
            lo, hi = DIM_RANGES[key]
            if not (lo <= val <= hi):
                ok = False
                break
        if not ok:
            continue

        corners_cv = blender_cam_to_opencv(corners_b)   # (8,3)
        bottom4 = corners_cv[BOTTOM_IDX]
        top4 = corners_cv[TOP_IDX]
        boxes_3d_cam = np.concatenate([bottom4, top4], axis=0)  # (8,3)

        center = boxes_3d_cam.mean(axis=0)
        depth = float(center[2])
        if not (MIN_DEPTH_M <= depth <= MAX_DEPTH_M):
            continue

        (px, py), w_px, h_px, angle = rbox_from_top_face(top4, fx, fy, cx, cy)
        if not (0.0 <= px <= width and 0.0 <= py <= height):
            continue
        area = w_px * h_px
        if area < MIN_RBOX_AREA_PX:
            continue

        objects.append(dict(
            name=p.get("name", ""),
            type=p.get("type", ""),
            object_dimensions_m=dict(width_m=width_m, length_m=length_m, height_m=height_m),
            boxes_3d_cam=boxes_3d_cam.round(6).tolist(),
            _area=area,
        ))

    objects.sort(key=lambda o: -o["_area"])
    objects = objects[:MAX_OBJECTS_PER_IMAGE]
    for o in objects:
        del o["_area"]

    if not objects:
        return None

    return dict(
        camera_intrinsics=dict(
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height,
            source_camera_model=ci.get("source_camera_model"),
            source_camera_type=ci.get("source_camera_type"),
        ),
        num_objects=len(objects),
        objects=objects,
    )


def main():
    label_paths = sorted(SRC_LABELS.glob("img_*.json"))
    print(f"Found {len(label_paths)} source labels in {SRC_LABELS}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_objects = 0
    skipped_no_image = 0
    obj_counts = []
    shard_dir = None

    for i, label_path in enumerate(label_paths):
        stem = label_path.stem                      # img_0000
        img_path = SRC_IMAGES / f"{stem}.png"
        if not img_path.exists():
            skipped_no_image += 1
            continue

        out = convert_one(label_path)
        if out is None:
            skipped_no_objects += 1
            continue

        if written % SHARD_SIZE == 0:
            shard_dir = OUT_ROOT / f"shard_{written:08d}"
            shard_dir.mkdir(parents=True, exist_ok=True)

        out_stem = f"{written:08d}"
        json_path = shard_dir / f"{out_stem}.json"
        png_path = shard_dir / f"{out_stem}.gray.png"

        json_path.write_text(json.dumps(out))
        if png_path.exists() or png_path.is_symlink():
            png_path.unlink()
        png_path.symlink_to(img_path.resolve())

        obj_counts.append(out["num_objects"])
        written += 1

    print(f"Wrote {written} samples to {OUT_ROOT}")
    print(f"Skipped: {skipped_no_image} (no image), {skipped_no_objects} (no valid objects after filtering)")
    if obj_counts:
        arr = np.array(obj_counts)
        print(f"Objects/image: min={arr.min()} max={arr.max()} mean={arr.mean():.1f} median={np.median(arr):.1f} total={arr.sum()}")


if __name__ == "__main__":
    main()
