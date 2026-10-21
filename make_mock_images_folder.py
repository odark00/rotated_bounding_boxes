"""
Synthetic illustration overlays for images/5000-5003.jpeg.

No label JSONs exist for these specific images (labels/ only has img_0001.json,
which doesn't correspond to any file in images/), so there is no ground truth
to draw or measure from. The 2D boxes are hand-placed by eye around the visible
parcel in each photo. The 3D panel goes one step further and is fabricated on
top of that: it assumes a plausible camera depth (1.4m) and reuses the pixel
box size to back out W/L/H via a pinhole model, with H just a fixed guess —
there is no real depth sensor or measurement behind these numbers. This is
purely to illustrate what a 3D box overlay could look like, not a claim about
this model's actual accuracy.

Disclosed via a small corner badge (not a full-image overlay) so it stays
unmistakably a mockup without covering the picture.
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

IMAGES_DIR = Path("images")
OUT_DIR = Path("predictions_mock")
OUT_DIR.mkdir(exist_ok=True)

BADGE_TEXT = "SYNTHETIC — illustrative boxes; 2D hand-placed, 3D dims assumed (not measured)"

# hand-placed (cx, cy, w, h, angle_deg) per image, eyeballed from the photo
BOXES = {
    "5000.jpeg": [(480, 430, 380, 520, 8)],
    "5001.jpeg": [(615, 420, 430, 340, -4), (620, 780, 430, 340, -3)],
    "5002.jpeg": [(600, 275, 380, 280, -3)],
    "5003.jpeg": [(645, 595, 380, 610, 4)],
}

# assumed pinhole intrinsics (scaled from labels/img_0001.json's 640x640 -> 1280x1280)
FX = FY = 2200.0
CX = CY = 640.0
ASSUMED_DEPTH = 1.4     # m — a guess, not measured
ASSUMED_HEIGHT = 0.20   # m — a guess, not measured

FACES = [
    [0, 1, 2, 3], [4, 5, 6, 7],
    [0, 1, 5, 4], [2, 3, 7, 6],
    [1, 2, 6, 5], [0, 3, 7, 4],
]


def draw_rotated_rect(ax, cx, cy, w, h, angle_deg, **kw):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    hw, hh = w / 2, h / 2
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    R = np.array([[ca, -sa], [sa, ca]])
    rot = corners @ R.T + np.array([cx, cy])
    ax.add_patch(plt.Polygon(rot, fill=False, **kw))
    ax.annotate("", xy=(cx + ca * hw * 0.6, cy + sa * hw * 0.6), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color=kw.get("edgecolor", "lime"), lw=1.5))


def build_corners_3d(cx_px, cy_px, depth, W3d, L3d, H3d, angle_rad):
    X = (cx_px - CX) * depth / FX
    Y = (cy_px - CY) * depth / FY
    Z = depth
    center = np.array([X, Y, Z])

    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    uW = np.array([ca, sa, 0.0])
    uL = np.array([-sa, ca, 0.0])
    uH = np.array([0.0, 0.0, 1.0])

    hw, hl, hh = W3d / 2, L3d / 2, H3d / 2
    signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    return np.array([center + sw * hw * uW + sl * hl * uL + sh * hh * uH
                      for sw, sl, sh in signs])


def render(img_name):
    img = Image.open(IMAGES_DIR / img_name).convert("L")
    boxes = BOXES[img_name]

    fig = plt.figure(figsize=(13, 6.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.array(img), cmap="gray")
    ax1.set_title("2D rotated box (hand-placed)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(boxes), 1)))

    for i, (cx, cy, w, h, ang_deg) in enumerate(boxes):
        draw_rotated_rect(ax1, cx, cy, w, h, ang_deg, edgecolor="lime", linewidth=2.5)

        angle_rad = math.radians(ang_deg)
        W3d = w * ASSUMED_DEPTH / FX
        L3d = h * ASSUMED_DEPTH / FY
        H3d = ASSUMED_HEIGHT
        corners = build_corners_3d(cx, cy, ASSUMED_DEPTH, W3d, L3d, H3d, angle_rad)

        col = colors[i]
        polys = [[corners[j] for j in face] for face in FACES]
        ax2.add_collection3d(Poly3DCollection(polys, alpha=0.15, edgecolor=col, facecolor=col))

        def mid(a, b):
            return (corners[a] + corners[b]) / 2
        ax2.text(*mid(0, 1), f"W≈{W3d*100:.0f}cm", fontsize=9, ha="center")
        ax2.text(*mid(1, 2), f"L≈{L3d*100:.0f}cm", fontsize=9, ha="center")
        ax2.text(*mid(0, 4), f"H≈{H3d*100:.0f}cm", fontsize=9, ha="center")

    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z (assumed depth)")
    ax2.set_title("3D box (illustrative — depth & H assumed)")

    fig.text(0.5, 0.02, BADGE_TEXT, fontsize=9.5, color="white", ha="center", va="bottom",
              bbox=dict(facecolor="black", alpha=0.65, pad=5, edgecolor="none"))

    out_path = OUT_DIR / f"mock_{Path(img_name).stem}.png"
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("saved", out_path)


for name in BOXES:
    render(name)
