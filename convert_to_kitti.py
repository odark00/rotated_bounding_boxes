"""
Convert the custom parcel dataset to KITTI object detection format.

Source structure:
  dataset/
    gray_images/img_XXXX.png       - 1000 training images (640x640)
    labels/img_XXXX.json           - 1000 training labels
    gray_images_test/img_XXXX.png  - 10 test images
    labels_test/img_XXXX.json      - 10 test labels

Output structure:
  dataset_kitti_format/object/
    ImageSets/train.txt, test.txt
    training/image_2/, label_2/, calib/
    testing/image_2/, calib/

Coordinate conventions
----------------------
Blender camera (source): +X right, +Y world-Y horizontal, -Z view direction (down).
  Objects in front have Z_b < 0, depth = -Z_b.
  Projection: u = fx * X_b / (-Z_b) + cx,  v = fy * (-Y_b) / (-Z_b) + cy

OCV camera: X_ocv = X_b, Y_ocv = -Y_b, Z_ocv = -Z_b (Z_ocv > 0 = depth below cam).

NEW KITTI coordinate system (rotated 90° around X from OCV):
  X_kitti = X_ocv           (same right axis)
  Y_kitti = Z_ocv           (depth → "down" = world vertical = physical height axis)
  Z_kitti = -Y_ocv          (horizontal → "forward")

  This makes roty(ry) rotate around Y_kitti = Z_ocv = world vertical axis, which
  is exactly the axis our floor-plane boxes rotate around.

Modified P2 for correct image projection in the new frame:
  u = fx * X_kitti / Y_kitti + cx  (Y_kitti = depth = Z_ocv)
  v = cy - fy * Z_kitti / Y_kitti  (Z_kitti = -Y_ocv)
  →  P2 = [[fx, cx, 0, 0], [0, cy, -fy, 0], [0, 1, 0, 0]]

Blender bound_box corner ordering (local space), edges from corner 0:
  0→1: local +Z  →  physical height (mapped to Y_kitti)
  0→3: local +Y  →  one horizontal dim (w, mapped to Z_kitti after ry rotation)
  0→4: local +X  →  other horizontal dim (l, the heading axis, mapped to X_kitti)

KITTI label:  type truncated occluded alpha left top right bottom h w l x y z ry
  h   = physical height (Y_kitti span = Z_ocv span)
  l   = local-X dimension (heading length, canonical KITTI X before ry rotation)
  w   = local-Y dimension (canonical KITTI Z before ry rotation)
  x   = mean X_ocv
  y   = max Z_ocv  (KITTI bottom = floor contact = farthest from camera)
  z   = mean(-Y_ocv) = mean Z_kitti
  ry  = atan2(-e4_b[1], e4_b[0])  where e4_b is edge 0→4 in Blender cam space
        (= heading angle, encodes the Blender Z-rotation of the box)
"""

import json
import math
import shutil
from pathlib import Path

SRC = Path("/home/daryna/code_ws/study/3D_detection/dataset")
DST = Path("/home/daryna/code_ws/study/3D_detection/dataset_kitti_format/object")

TRAIN_IMG_DIR   = SRC / "gray_images"
TRAIN_LABEL_DIR = SRC / "labels"
TEST_IMG_DIR    = SRC / "gray_images_test"
TEST_LABEL_DIR  = SRC / "labels_test"

OBJ_CLASS = "Parcel"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def norm(v):
    return math.sqrt(sum(x * x for x in v))


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def blender_to_ocv(corners_blender):
    """Blender camera space → OpenCV (KITTI) camera space."""
    return [[X, -Y, -Z] for X, Y, Z in corners_blender]


def project_ocv(X, Y, Z, fx, fy, cx, cy):
    """Standard pinhole projection in OpenCV camera space (Z > 0)."""
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u, v


def bbox2d(corners_ocv, fx, fy, cx, cy, img_w, img_h):
    """Tight 2D KITTI bbox [left, top, right, bottom] from 8 OCV corners."""
    us, vs = [], []
    for X, Y, Z in corners_ocv:
        if Z <= 0:
            return None
        u, v = project_ocv(X, Y, Z, fx, fy, cx, cy)
        us.append(u)
        vs.append(v)
    left   = max(0.0,       min(us))
    top    = max(0.0,       min(vs))
    right  = min(float(img_w), max(us))
    bottom = min(float(img_h), max(vs))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def box3d_params(corners_blender):
    """
    Return (h, w, l, cx, cy, cz, ry) in the NEW KITTI coordinate system.

    New system: X_kitti=X_ocv, Y_kitti=Z_ocv (depth), Z_kitti=-Y_ocv.
    roty(ry) now rotates around Y_kitti = Z_ocv = world vertical axis, which
    is the axis boxes actually rotate around on the floor.

    Edges from corner 0 (Blender bound_box ordering):
      e1 = c[1]-c[0] : local +Z  →  physical height (maps to Y_kitti)
      e3 = c[3]-c[0] : local +Y  →  horizontal w direction
      e4 = c[4]-c[0] : local +X  →  horizontal l direction (heading axis)

    ry = atan2(-e4_b[1], e4_b[0])
      Because roty(ry) @ (1,0,0) = (cos ry, 0, -sin ry) must equal
      the unit local-X direction in new coords = (e4[0], 0, e4[1]) / |e4|,
      giving cos(ry)=e4[0]/|e4|, -sin(ry)=e4[1]/|e4| → ry = atan2(-e4[1], e4[0]).

    Center (cx, cy, cz) in the new coordinate system:
      cx = mean X_ocv      (new X)
      cy = max  Z_ocv      (new Y, KITTI bottom = floor contact = max depth)
      cz = mean(-Y_ocv)    (new Z)
    """
    c = corners_blender

    e1 = sub(c[1], c[0])   # height direction (Blender cam Z, world vertical)
    e3 = sub(c[3], c[0])   # local Y horizontal dim
    e4 = sub(c[4], c[0])   # local X horizontal dim (heading)

    h = norm(e1)
    l = norm(e4)
    w = norm(e3)

    # ry from heading edge in the new (X_kitti, Z_kitti) = (X_ocv, -Y_ocv) floor plane
    ry = math.atan2(-e4[1], e4[0])
    # Normalize to (-pi, pi]
    ry = (ry + math.pi) % (2 * math.pi) - math.pi

    corners_ocv = blender_to_ocv(c)
    xs = [p[0] for p in corners_ocv]
    ys = [p[1] for p in corners_ocv]
    zs = [p[2] for p in corners_ocv]

    cx = (max(xs) + min(xs)) / 2
    cy = max(zs)                     # new Y = max Z_ocv = KITTI bottom (floor contact)
    cz = -(max(ys) + min(ys)) / 2   # new Z = -mean Y_ocv

    return h, w, l, cx, cy, cz, ry


# ---------------------------------------------------------------------------
# Calibration file
# ---------------------------------------------------------------------------

def make_calib_lines(fx, fy, cx_px, cy_px):
    """
    KITTI calib.txt content.

    P2 uses the modified projection matrix for the new coordinate system
    (X_kitti=X_ocv, Y_kitti=Z_ocv, Z_kitti=-Y_ocv):

      u = fx*X_kitti/Y_kitti + cx   (Y_kitti is the depth divisor)
      v = cy - fy*Z_kitti/Y_kitti

      P2 = [[fx, cx, 0, 0], [0, cy, -fy, 0], [0, 1, 0, 0]]

    This correctly projects 3D points in the new system to image pixels.
    R0_rect, Tr_velo_to_cam, Tr_imu_to_velo: identity (no LiDAR/IMU).
    """
    P = (f"{fx:.6e} {cx_px:.6e} 0.000000e+00 0.000000e+00 "
         f"0.000000e+00 {cy_px:.6e} {-fy:.6e} 0.000000e+00 "
         f"0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00")
    R0 = "1.000000e+00 0.000000e+00 0.000000e+00 " \
         "0.000000e+00 1.000000e+00 0.000000e+00 " \
         "0.000000e+00 0.000000e+00 1.000000e+00"
    Tr = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 " \
         "0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 " \
         "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
    return "\n".join([
        f"P0: {P}",
        f"P1: {P}",
        f"P2: {P}",
        f"P3: {P}",
        f"R0_rect: {R0}",
        f"Tr_velo_to_cam: {Tr}",
        f"Tr_imu_to_velo: {Tr}",
    ]) + "\n"


# ---------------------------------------------------------------------------
# Label file
# ---------------------------------------------------------------------------

def make_label_lines(label_json):
    intr   = label_json["camera_intrinsics"]
    fx, fy = intr["fx"], intr["fy"]
    cx_px  = intr["cx"]
    cy_px  = intr["cy"]
    img_w  = intr["width"]
    img_h  = intr["height"]

    lines = []
    for parcel in label_json.get("parcels", []):
        corners_b = parcel["bbox_3d_camera"]

        # Skip boxes behind the camera
        if any(z >= 0 for _, _, z in corners_b):
            continue

        corners_ocv = blender_to_ocv(corners_b)

        bb = bbox2d(corners_ocv, fx, fy, cx_px, cy_px, img_w, img_h)
        if bb is None:
            continue

        left, top, right, bottom = bb

        h, w, l, cx, cy, cz, ry = box3d_params(corners_b)

        if any(d <= 0 for d in (h, w, l)):
            continue

        # alpha: observation angle — azimuth uses (cx, cy) where cy = depth (Y_kitti)
        alpha = ry - math.atan2(cx, cy)
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

        line = (
            f"{OBJ_CLASS} "
            f"0.00 "           # truncated
            f"0 "              # occluded
            f"{alpha:.2f} "
            f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
            f"{h:.4f} {w:.4f} {l:.4f} "
            f"{cx:.4f} {cy:.4f} {cz:.4f} "
            f"{ry:.4f}"
        )
        lines.append(line)

    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_split(img_dir, label_dir, out_img_dir, out_calib_dir,
                  out_label_dir, frame_id_start=0):
    """
    Convert one split (training or testing).

    Returns the list of 6-digit frame ID strings written.
    """
    label_paths = sorted(label_dir.glob("*.json")) if label_dir else []
    img_paths   = sorted(img_dir.glob("*.png"))

    # Use label_paths as the primary source; fall back to images for test set
    sources = label_paths if label_paths else img_paths

    frame_ids = []
    for offset, src_path in enumerate(sources):
        fid = f"{frame_id_start + offset:06d}"
        stem = src_path.stem  # e.g. "img_0003"

        # Image
        src_img = img_dir / f"{stem}.png"
        if src_img.exists():
            shutil.copy2(src_img, out_img_dir / f"{fid}.png")

        # Calibration
        if label_dir and (label_dir / f"{stem}.json").exists():
            lbl_json = json.loads((label_dir / f"{stem}.json").read_text())
            intr     = lbl_json["camera_intrinsics"]
            calib_txt = make_calib_lines(intr["fx"], intr["fy"],
                                         intr["cx"], intr["cy"])
        else:
            # Default intrinsics from the Blender script
            calib_txt = make_calib_lines(1100.0, 1100.0, 320.0, 320.0)

        (out_calib_dir / f"{fid}.txt").write_text(calib_txt)

        # Labels (training only; KITTI test set has no public labels)
        if out_label_dir is not None and label_dir is not None:
            lbl_json   = json.loads((label_dir / f"{stem}.json").read_text())
            label_text = make_label_lines(lbl_json)
            (out_label_dir / f"{fid}.txt").write_text(label_text)

        frame_ids.append(fid)

    return frame_ids


def main():
    # Create output directories
    for split in ("training", "testing"):
        for sub in ("image_2", "calib"):
            (DST / split / sub).mkdir(parents=True, exist_ok=True)
    (DST / "training" / "label_2").mkdir(parents=True, exist_ok=True)
    (DST / "ImageSets").mkdir(parents=True, exist_ok=True)

    print("Converting training split …")
    train_ids = convert_split(
        img_dir       = TRAIN_IMG_DIR,
        label_dir     = TRAIN_LABEL_DIR,
        out_img_dir   = DST / "training" / "image_2",
        out_calib_dir = DST / "training" / "calib",
        out_label_dir = DST / "training" / "label_2",
        frame_id_start = 0,
    )
    print(f"  {len(train_ids)} frames written.")

    print("Converting testing split …")
    test_ids = convert_split(
        img_dir       = TEST_IMG_DIR,
        label_dir     = TEST_LABEL_DIR,
        out_img_dir   = DST / "testing" / "image_2",
        out_calib_dir = DST / "testing" / "calib",
        out_label_dir = None,       # KITTI test set has no labels
        frame_id_start = 0,
    )
    print(f"  {len(test_ids)} frames written.")

    # ImageSets
    (DST / "ImageSets" / "train.txt").write_text("\n".join(train_ids) + "\n")
    (DST / "ImageSets" / "test.txt").write_text("\n".join(test_ids) + "\n")
    print("ImageSets written.")

    # Sanity-check one training label
    sample = (DST / "training" / "label_2" / f"{train_ids[0]}.txt").read_text()
    print(f"\nSample label ({train_ids[0]}.txt):\n{sample.strip()}")


if __name__ == "__main__":
    main()
