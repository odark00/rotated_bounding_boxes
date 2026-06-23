"""
Ground Truth 3D Bounding Box Visualizer
Loads a JSON label + corresponding image and overlays oriented 3D bboxes.

Usage:
    python visualize_gt.py --image img_0000.png --label img_0000.json
    python visualize_gt.py --image_dir .../images --label_dir .../labels --index 0
    python visualize_gt.py --image_dir .../images --label_dir .../labels --all --out_dir viz/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from PIL import Image


# 12 edges of a cube (corner index pairs)
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # face 1 (near in object frame)
    (4, 5), (5, 6), (6, 7), (7, 4),   # face 2 (far)
    (0, 4), (1, 5), (2, 6), (3, 7),   # connecting edges
]

# 6 faces of a cube (corner indices)
FACES = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 2, 6, 5],
    [0, 3, 7, 4],
]

# distinct colors for boxes
COLORS = [
    "#00ff7f", "#ff3366", "#33aaff", "#ffaa00",
    "#aa33ff", "#00ffff", "#ff66cc", "#ccff33",
    "#ff8844", "#44ff88",
]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def project_points(points_cam, K, blender_convention=True):
    """
    Project 3D camera-frame points to 2D pixels.
    Blender camera: looks down -Z, so depth = -z.
    Standard pinhole: looks down +Z, so depth = +z.
    """
    pts = points_cam.copy().astype(np.float64)
    if blender_convention:
        # Blender: x right, y up, -z forward.  Convert to OpenCV-like (x right, y down, +z forward)
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

    z = np.clip(pts[:, 2:3], 1e-6, None)
    xy = pts[:, :2] / z
    uv = xy @ K[:2, :2].T + K[:2, 2]
    return uv, pts[:, 2]  # (N,2), depth


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_box_2d(ax, corners_2d, depths, color, name=None, score=None):
    """Draw 12 edges of a 3D bbox projected onto 2D image."""
    # split edges: visible (front face) vs occluded (back) by depth ordering
    mean_depths = []
    for a, b in EDGES:
        mean_depths.append((depths[a] + depths[b]) / 2)
    mean_depths = np.array(mean_depths)
    median = np.median(mean_depths)

    for i, (a, b) in enumerate(EDGES):
        x = [corners_2d[a, 0], corners_2d[b, 0]]
        y = [corners_2d[a, 1], corners_2d[b, 1]]
        if mean_depths[i] > median:
            # farther edges -> dashed, thinner
            ax.plot(x, y, color=color, linewidth=1.0, linestyle="--", alpha=0.7)
        else:
            ax.plot(x, y, color=color, linewidth=2.2, alpha=0.95)

    # draw corners
    ax.scatter(corners_2d[:, 0], corners_2d[:, 1], s=12, c=color,
               edgecolors="black", linewidths=0.5, zorder=5)

    # label at top-most corner
    top_idx = int(np.argmin(corners_2d[:, 1]))
    label = name or ""
    if score is not None:
        label += f" {score:.2f}"
    if label:
        ax.text(corners_2d[top_idx, 0], corners_2d[top_idx, 1] - 6,
                label, color="white", fontsize=8,
                bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"))


def draw_box_3d(ax, corners_3d, color, name=None):
    """Draw 3D bbox as faces + edges in 3D matplotlib axes."""
    polys = [[corners_3d[i] for i in f] for f in FACES]
    pc = Poly3DCollection(polys, alpha=0.15, facecolor=color,
                          edgecolor=color, linewidth=1.5)
    ax.add_collection3d(pc)
    # corners
    ax.scatter(corners_3d[:, 0], corners_3d[:, 1], corners_3d[:, 2],
               s=18, c=color, edgecolors="black", linewidths=0.4)
    if name:
        c = corners_3d.mean(axis=0)
        ax.text(c[0], c[1], c[2], name, color=color, fontsize=8)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_label(label_path):
    with open(label_path, "r") as f:
        return json.load(f)


def get_intrinsics(label, target_size=None, original_size=None):
    """Build K matrix.  If image was resized, rescale fx,fy,cx,cy."""
    c = label["camera_intrinsics"]
    K = np.array([
        [c["fx"], 0,      c["cx"]],
        [0,       c["fy"], c["cy"]],
        [0,       0,      1.0],
    ], dtype=np.float64)
    if target_size and original_size:
        ow, oh = original_size
        tw, th = target_size
        sx, sy = tw / ow, th / oh
        K[0] *= sx
        K[1] *= sy
    return K


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------
def visualize(image_path, label_path, out_path=None, show=True,
              draw_camera_axes=True, show_3d=True):
    label = load_label(label_path)
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)
    H, W = img_np.shape

    # The JSON intrinsics already match the rendered image size, no rescale needed
    K = get_intrinsics(label)

    parcels = label["parcels"]
    n = len(parcels)
    print(f"\n[{Path(image_path).name}] num_parcels={n}")

    # ---------- Build figure ----------
    if show_3d:
        fig = plt.figure(figsize=(16, 7))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        fig, ax_img = plt.subplots(1, 1, figsize=(9, 9))
        ax_3d = None

    ax_img.imshow(img_np, cmap="gray")
    ax_img.set_title(f"{Path(image_path).name} — {n} parcels", fontsize=11)
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.axis("off")

    all_corners_3d = []
    for i, p in enumerate(parcels):
        color = COLORS[i % len(COLORS)]
        corners_cam = np.array(p["bbox_3d_camera"], dtype=np.float64)  # (8,3)
        all_corners_3d.append(corners_cam)

        # ----- 2D projection -----
        uv, depth = project_points(corners_cam, K, blender_convention=True)
        draw_box_2d(ax_img, uv, depth, color=color, name=p["name"])

        # log
        center = corners_cam.mean(axis=0)
        d01 = np.linalg.norm(corners_cam[0] - corners_cam[1])
        d12 = np.linalg.norm(corners_cam[1] - corners_cam[2])
        d04 = np.linalg.norm(corners_cam[0] - corners_cam[4])
        print(f"  [{i}] {p['name']:14s}  center={center.round(3).tolist()}  "
              f"size=({d01:.3f},{d12:.3f},{d04:.3f})  "
              f"rot={[round(r,3) for r in p['rotation']]}")

        # ----- 3D -----
        if ax_3d is not None:
            draw_box_3d(ax_3d, corners_cam, color=color, name=p["name"])

    # ----- 3D scene decoration -----
    if ax_3d is not None and len(all_corners_3d):
        all_pts = np.concatenate(all_corners_3d, axis=0)

        # camera at origin
        ax_3d.scatter([0], [0], [0], s=80, c="red", marker="^", label="Camera")
        if draw_camera_axes:
            L = 0.15
            ax_3d.plot([0, L], [0, 0], [0, 0], color="red")
            ax_3d.plot([0, 0], [0, L], [0, 0], color="green")
            ax_3d.plot([0, 0], [0, 0], [0, -L], color="blue")  # -Z forward

        ax_3d.set_xlabel("X (m)")
        ax_3d.set_ylabel("Y (m)")
        ax_3d.set_zlabel("Z (m)")
        ax_3d.set_title("3D bboxes (camera frame, -Z forward)")

        # equal aspect-ish
        mins = np.minimum(all_pts.min(0), np.array([0, 0, -0.05]))
        maxs = np.maximum(all_pts.max(0), np.array([0, 0,  0.05]))
        ranges = maxs - mins
        r = ranges.max() / 2
        mid = (maxs + mins) / 2
        ax_3d.set_xlim(mid[0] - r, mid[0] + r)
        ax_3d.set_ylim(mid[1] - r, mid[1] + r)
        ax_3d.set_zlim(mid[2] - r, mid[2] + r)
        ax_3d.legend(loc="upper right", fontsize=8)
        ax_3d.view_init(elev=20, azim=-60)

    plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"  saved -> {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, help="single image path")
    p.add_argument("--label", type=str, help="single label path")
    p.add_argument("--image_dir", type=str,
                   default="/Volumes/playground/it_cl_terminal/ilo/data/blender_generated/images")
    p.add_argument("--label_dir", type=str,
                   default="/Volumes/playground/it_cl_terminal/ilo/data/blender_generated/labels")
    p.add_argument("--index", type=int, default=0, help="index into sorted images")
    p.add_argument("--name", type=str, default=None,
                   help="filename stem, e.g. img_0003 (overrides --index)")
    p.add_argument("--all", action="store_true", help="render all samples")
    p.add_argument("--out_dir", type=str, default=None,
                   help="save figures here (no interactive show)")
    p.add_argument("--no_3d", action="store_true", help="skip 3D subplot")
    args = p.parse_args()

    show_3d = not args.no_3d

    # ---- single explicit pair ----
    if args.image and args.label:
        out = None
        if args.out_dir:
            out = Path(args.out_dir) / (Path(args.image).stem + "_gt.png")
        visualize(args.image, args.label, out_path=out,
                  show=args.out_dir is None, show_3d=show_3d)
        return

    img_dir = Path(args.image_dir)
    lbl_dir = Path(args.label_dir)
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    if not stems:
        raise FileNotFoundError(f"No PNGs found in {img_dir}")

    # ---- all ----
    if args.all:
        for stem in stems:
            img = img_dir / f"{stem}.png"
            lbl = lbl_dir / f"{stem}.json"
            if not lbl.exists():
                print(f"skip {stem} (no label)")
                continue
            out = None
            if args.out_dir:
                out = Path(args.out_dir) / f"{stem}_gt.png"
            visualize(img, lbl, out_path=out,
                      show=args.out_dir is None, show_3d=show_3d)
        return

    # ---- single by name/index ----
    stem = args.name if args.name else stems[args.index]
    img = img_dir / f"{stem}.png"
    lbl = lbl_dir / f"{stem}.json"
    out = Path(args.out_dir) / f"{stem}_gt.png" if args.out_dir else None
    visualize(img, lbl, out_path=out,
              show=args.out_dir is None, show_3d=show_3d)


if __name__ == "__main__":
    main()