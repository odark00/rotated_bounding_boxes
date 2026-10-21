"""
Generates CLEARLY-LABELED synthetic illustration images (not model output).

Renders ground-truth boxes from TASLER_MULTI in the same visual style as
predict.py's overlay (2D rotated boxes + 3D bounding boxes), so they can be
used to illustrate "what good detection would look like" — every image is
watermarked as synthetic/target illustration so it can never be mistaken for
a real evaluation result.
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

FACES = [
    [0, 1, 2, 3], [4, 5, 6, 7],
    [0, 1, 5, 4], [2, 3, 7, 6],
    [1, 2, 6, 5], [0, 3, 7, 4],
]

SAMPLES = [
    ("TASLER_MULTI/shard_00001500/00001519.json", 1),
    ("TASLER_MULTI/shard_00001250/00001447.json", 6),
    ("TASLER_MULTI/shard_00000000/00000147.json", 12),
    ("TASLER_MULTI/shard_00000000/00000050.json", 24),
]

OUT_DIR = Path("predictions_mock")
OUT_DIR.mkdir(exist_ok=True)

WATERMARK = "SYNTHETIC — TARGET ILLUSTRATION, NOT MODEL OUTPUT"
CAPTION = "Ground-truth boxes rendered for illustration only — not an inference result"


def project(pts_cam, fx, fy, cx, cy):
    z = np.clip(pts_cam[:, 2:3], 1e-3, None)
    xy = pts_cam[:, :2] / z
    return xy * np.array([[fx, fy]]) + np.array([[cx, cy]])


def add_watermark(fig):
    fig.text(0.5, 0.5, WATERMARK, fontsize=26, color="red", alpha=0.35,
              ha="center", va="center", rotation=28, fontweight="bold",
              zorder=10)
    fig.text(0.5, 0.015, CAPTION, fontsize=10, color="#8a1f1f",
              ha="center", va="bottom", style="italic")


def render(json_path, out_path):
    meta = json.loads(Path(json_path).read_text())
    img_path = Path(json_path).with_suffix("").with_suffix(".gray.png")
    img = Image.open(img_path).convert("L")

    K = meta["camera_intrinsics"]
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.array(img), cmap="gray")
    ax1.set_title("2D rotated boxes (ground truth)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    N = len(meta["objects"])
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(N, 1)))

    for i, obj in enumerate(meta["objects"]):
        corners_cam = np.array(obj["boxes_3d_cam"])  # (8,3)

        # ---- 2D overlay: project the near face (closer to camera) ----
        near_face = corners_cam[4:8] if corners_cam[4, 2] < corners_cam[0, 2] else corners_cam[0:4]
        uv = project(near_face, fx, fy, cx, cy)
        poly = plt.Polygon(uv, fill=False, edgecolor="lime", linewidth=2)
        ax1.add_patch(poly)
        centroid = uv.mean(axis=0)
        fwd = uv[1] - uv[0]
        fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
        ax1.annotate("", xy=centroid + fwd * 18, xytext=centroid,
                      arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))

        # ---- 3D panel ----
        col = colors[i]
        polys = [[corners_cam[j] for j in face] for face in FACES]
        ax2.add_collection3d(Poly3DCollection(polys, alpha=0.15,
                                               edgecolor=col, facecolor=col))
        if N <= 6:
            d = obj["object_dimensions_m"]

            def mid(a, b):
                return (corners_cam[a] + corners_cam[b]) / 2
            ax2.text(*mid(0, 1), f"W={d['width_m']*100:.1f}cm", fontsize=9, ha="center")
            ax2.text(*mid(1, 2), f"L={d['length_m']*100:.1f}cm", fontsize=9, ha="center")
            ax2.text(*mid(0, 4), f"H={d['height_m']*100:.1f}cm", fontsize=9, ha="center")

    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z (depth)")
    ax2.set_title("3D bounding boxes (ground truth)")

    add_watermark(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}  ({N} objects)")


for i, (json_path, n) in enumerate(SAMPLES):
    out_path = OUT_DIR / f"mock_illustration_{i}_n{n}.png"
    render(json_path, out_path)
