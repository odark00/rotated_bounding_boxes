"""
Inference script for RTMDet-Rotated parcel detector.

Usage:
    python predict.py --ckpt model.ckpt --image img.png
    python predict.py --ckpt model.ckpt --image img.png \\
        --intrinsics 1100 1100 320 320   # fx fy cx cy of original image

Output per detection:
    score, cx_px, cy_px, w_px, h_px, angle_deg,
    depth_m,  W_3d_m, L_3d_m, H_3d_m
"""

import argparse
import math
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from losses import RTMDetLitModule

STRIDES = [8, 16, 32]

FACES = [
    [0,1,2,3], [4,5,6,7],
    [0,1,5,4], [2,3,7,6],
    [1,2,6,5], [0,3,7,4],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_level(out, stride, score_thresh):
    """Decode one FPN level → list of raw detections."""
    B, _, H, W = out['cls'].shape
    assert B == 1

    scores = out['cls'][0, 0].sigmoid().flatten()        # (M,)
    keep   = scores > score_thresh
    if keep.sum() == 0:
        return None

    def _g(k): return out[k][0, 0].flatten()[keep].cpu().numpy()

    return dict(
        scores =scores[keep].cpu().numpy(),
        cx     =_g('cx'),  cy    =_g('cy'),
        w      =_g('w'),   h     =_g('h'),
        sin_a  =_g('sin_a'), cos_a=_g('cos_a'),
        depth  =_g('depth'),
        height3d=_g('height3d'),
    )


def nms_rotated(dets, iou_thresh=0.5):
    """Simple centre-distance NMS (avoids rotated-IoU complexity)."""
    if len(dets['scores']) == 0:
        return dets

    idx  = np.argsort(-dets['scores'])
    keep = []
    used = np.zeros(len(idx), bool)
    cx, cy = dets['cx'], dets['cy']
    ww, hh = dets['w'], dets['h']

    for ii, i in enumerate(idx):
        if used[ii]:
            continue
        keep.append(i)
        for jj, j in enumerate(idx[ii+1:], ii+1):
            if used[jj]:
                continue
            dist   = math.hypot(cx[i]-cx[j], cy[i]-cy[j])
            radius = 0.5 * (max(ww[i], hh[i]) + max(ww[j], hh[j]))
            if dist < radius * iou_thresh:
                used[jj] = True

    return {k: v[keep] for k, v in dets.items()}


def build_corners_3d(cx_px, cy_px, depth, W3d, L3d, H3d, angle, fx, fy, cx_K, cy_K):
    """
    Reconstruct 8 corners of the 3D box in image-space 3D
    (x right, y down, z forward — OpenCV convention).

    angle : rotation of the W-axis in the image plane (radians).
    """
    X = (cx_px - cx_K) * depth / fx
    Y = (cy_px - cy_K) * depth / fy
    Z = depth
    center = np.array([X, Y, Z])

    ca, sa = math.cos(angle), math.sin(angle)
    # W axis: along angle in image plane (horizontal)
    uW = np.array([ ca,  sa, 0.0])
    # L axis: perpendicular to W in image plane
    uL = np.array([-sa,  ca, 0.0])
    # H axis: depth direction (into scene)
    uH = np.array([0.0, 0.0, 1.0])

    hw, hl, hh = W3d/2, L3d/2, H3d/2
    signs = [(-1,-1,-1),( 1,-1,-1),( 1, 1,-1),(-1, 1,-1),
             (-1,-1, 1),( 1,-1, 1),( 1, 1, 1),(-1, 1, 1)]
    corners = np.array([center + sw*hw*uW + sl*hl*uL + sh*hh*uH
                        for sw, sl, sh in signs])
    return corners


def project_to_image(corners_3d, fx, fy, cx_K, cy_K):
    """Project (N,3) OpenCV-frame 3D points to pixel (N,2)."""
    z   = np.clip(corners_3d[:, 2:3], 1e-3, None)
    xy  = corners_3d[:, :2] / z
    uv  = xy * np.array([[fx, fy]]) + np.array([[cx_K, cy_K]])
    return uv


def draw_rotated_rect(ax, cx, cy, w, h, angle_rad, **kw):
    """Draw an oriented rectangle on a matplotlib axis."""
    ca, sa  = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh  = w / 2, h / 2
    corners = np.array([
        [-hw, -hh], [ hw, -hh], [ hw,  hh], [-hw,  hh]
    ])
    R     = np.array([[ca, -sa], [sa, ca]])
    rotc  = corners @ R.T + np.array([cx, cy])
    poly  = plt.Polygon(rotc, fill=False, **kw)
    ax.add_patch(poly)
    # draw centre direction arrow
    ax.annotate('', xy=(cx + ca*hw*0.6, cy + sa*hw*0.6),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=kw.get('edgecolor','lime'), lw=1.5))


# ---------------------------------------------------------------------------
# Main predict
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(ckpt, image_path, conf_thresh=0.5, intrinsics=None, out_path="prediction.png"):

    model = RTMDetLitModule.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()

    img    = Image.open(image_path).convert("L")
    W0, H0 = img.size
    tfm    = T.Compose([T.Resize((640, 640)), T.ToTensor(), T.Normalize([0.5],[0.5])])
    x      = tfm(img).unsqueeze(0)

    # ---- intrinsics ----
    if intrinsics is not None:
        fx_orig, fy_orig, cx_orig, cy_orig = intrinsics
        sx, sy = 640.0/W0, 640.0/H0
        fx_m, fy_m = fx_orig*sx, fy_orig*sy
        cx_m, cy_m = cx_orig*sx, cy_orig*sy
    else:
        fx_m = fy_m = 1100.0
        cx_m = cy_m = 320.0
        fx_orig = fy_orig = 1100.0
        cx_orig, cy_orig = W0/2, H0/2

    level_outs = model.model(x)

    # ---- decode all levels ----
    all_dets = {k: [] for k in
                ['scores','cx','cy','w','h','sin_a','cos_a','depth','height3d']}
    for out, stride in zip(level_outs, STRIDES):
        d = decode_level(out, stride, conf_thresh)
        if d is not None:
            for k in all_dets:
                all_dets[k].append(d[k])

    if not all_dets['scores']:
        print("No parcels detected above threshold.")
        return

    for k in all_dets:
        all_dets[k] = np.concatenate(all_dets[k])

    dets = nms_rotated(all_dets)
    N = len(dets['scores'])

    # ---- 3D dimensions using intrinsics ----
    W3d  = dets['w']      * dets['depth'] / fx_m
    L3d  = dets['h']      * dets['depth'] / fy_m
    H3d  = dets['height3d']
    angles = np.arctan2(dets['sin_a'], dets['cos_a'])
    angles[angles < 0] += math.pi    # canonicalise to [0, π)

    print(f"\nDetected {N} parcel(s):")
    for i in range(N):
        print(f"  [{i}] score={dets['scores'][i]:.3f}"
              f"  cx={dets['cx'][i]:.1f}px  cy={dets['cy'][i]:.1f}px"
              f"  angle={math.degrees(angles[i]):.1f}°"
              f"  depth={dets['depth'][i]:.4f}m"
              f"  W={W3d[i]:.4f}m  L={L3d[i]:.4f}m  H={H3d[i]:.4f}m")

    # ---- visualise ----
    fig = plt.figure(figsize=(14, 6))

    # --- 2D rotated boxes on image ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.array(img), cmap="gray")
    # rescale model-space coords to original image resolution
    sx_v, sy_v = W0/640.0, H0/640.0
    for i in range(N):
        draw_rotated_rect(ax1,
            dets['cx'][i]*sx_v, dets['cy'][i]*sy_v,
            dets['w'][i]*sx_v,  dets['h'][i]*sy_v,
            angles[i],
            edgecolor='lime', linewidth=2,
        )
        ax1.text(dets['cx'][i]*sx_v, dets['cy'][i]*sy_v - dets['h'][i]*sy_v*0.6,
                 f"{dets['scores'][i]:.2f}", color='yellow', fontsize=8, ha='center')
    ax1.set_title("2D rotated boxes")
    ax1.axis("off")

    # --- 3D boxes ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for i in range(N):
        corners = build_corners_3d(
            dets['cx'][i], dets['cy'][i], dets['depth'][i],
            W3d[i], L3d[i], H3d[i], angles[i],
            fx_m, fy_m, cx_m, cy_m,
        )
        # project to original image for 3D-on-image overlay (optional)
        polys = [[corners[j] for j in face] for face in FACES]
        ax2.add_collection3d(Poly3DCollection(polys, alpha=0.2,
                                               edgecolor='k', facecolor='cyan'))
        ax2.scatter(corners[:,0], corners[:,1], corners[:,2], s=10)

    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z (depth)")
    ax2.set_title("3D boxes (OpenCV frame, Z forward)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--conf",  type=float, default=0.5)
    p.add_argument("--out",   default="prediction.png")
    p.add_argument("--intrinsics", type=float, nargs=4,
                   metavar=("FX","FY","CX","CY"),
                   help="Camera intrinsics of the input image. "
                        "Enables metric W/L/H estimation.")
    args = p.parse_args()
    predict(args.ckpt, args.image, args.conf,
            intrinsics=tuple(args.intrinsics) if args.intrinsics else None,
            out_path=args.out)
