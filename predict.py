"""
Inference script for RTMDet-Rotated parcel detector.

Usage:
    python predict.py --ckpt model.ckpt --image img.png --intrinsics FX FY CX CY
    python predict.py --ckpt model.ckpt --folder imgs/ --out_dir predictions/ \
        --intrinsics 1100 1100 320 320   # fx fy cx cy of the ORIGINAL image

Output per detection:
    score, cx_px, cy_px, w_px, h_px, angle_deg,
    depth_m,  W_3d_m, L_3d_m, H_3d_m

`--intrinsics` is not cosmetic: the network predicts intrinsics-free quantities
(canonical depth and apparent pixel height, see model.py) and the metric answer
is obtained by scaling them with fx/fy. Without it the script assumes the
checkpoint's reference camera, which is only right if the image happens to come
from a lens with fx == fx_ref.
"""

import argparse
import math
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from losses import RTMDetLitModule
from model import decode_3d, RTMDetRotated
from geometry import rotated_iou

STRIDES = [8, 16, 32]

# every per-anchor head output except the raw objectness logit, plus its score
DET_KEYS = ['scores'] + [k for k in RTMDetRotated.OUT_KEYS if k != 'cls']

FACES = [
    [0,1,2,3], [4,5,6,7],
    [0,1,5,4], [2,3,7,6],
    [1,2,6,5], [0,3,7,4],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_level(out, stride, score_thresh):
    """Decode one FPN level -> dict of raw detections (or None)."""
    B, _, H, W = out['cls'].shape
    assert B == 1

    scores = out['cls'][0, 0].sigmoid().flatten()        # (M,)
    keep   = scores > score_thresh
    if keep.sum() == 0:
        return None

    def _g(k): return out[k][0, 0].flatten()[keep].float().cpu().numpy()

    d = {k: _g(k) for k in DET_KEYS if k != 'scores'}
    d['scores'] = scores[keep].float().cpu().numpy()
    return d


def angles_from(dets):
    """
    Recover the box angle in [0, pi) from the head's (sin 2a, cos 2a) pair.

    The head predicts the doubled angle (losses.py) so that the 180-degree
    rectangle symmetry is continuous; halving atan2's (-pi, pi] output gives
    (-pi/2, pi/2], which is then wrapped into [0, pi).
    """
    a = 0.5 * np.arctan2(dets['sin_a'], dets['cos_a'])
    return np.mod(a, math.pi)


def nms_rotated(dets, iou_thresh=0.3):
    """
    Greedy NMS on true rotated-box IoU.

    The previous version approximated overlap with a centre-distance test,
    which merges distinct parcels that sit close together (common here: images
    hold up to 30 packed packages) while leaving genuine duplicates of a long
    thin box alive. `geometry.rotated_iou` does the real polygon clip and
    short-circuits on a circumradius test, so this stays cheap.
    """
    n = len(dets['scores'])
    if n == 0:
        return dets

    angles = angles_from(dets)
    boxes = [(dets['cx'][i], dets['cy'][i], dets['w'][i], dets['h'][i], angles[i])
             for i in range(n)]

    order = np.argsort(-dets['scores'])
    suppressed = np.zeros(n, bool)
    keep = []
    for ii, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order[ii + 1:]:
            if not suppressed[j] and rotated_iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed[j] = True

    return {k: v[keep] for k, v in dets.items()}


def detect(model, x, conf_thresh=0.5, nms_iou=0.3):
    """Run the model on a (1,1,S,S) tensor and return NMS'd detections."""
    level_outs = model.model(x)
    all_dets = {k: [] for k in DET_KEYS}
    for out, stride in zip(level_outs, STRIDES):
        d = decode_level(out, stride, conf_thresh)
        if d is not None:
            for k in all_dets:
                all_dets[k].append(d[k])
    if not all_dets['scores']:
        return None
    return nms_rotated({k: np.concatenate(v) for k, v in all_dets.items()}, nms_iou)


def build_corners_3d(cx_px, cy_px, depth, W3d, L3d, H3d, angle, fx, fy, cx_K, cy_K):
    """
    Reconstruct 8 corners of the 3D box in camera-frame 3D
    (x right, y down, z forward — OpenCV convention).

    angle : image-plane direction of the box's LONG edge (radians), so the W3d
            axis is the long one (see dataset._rbox_from_top_face).
    """
    X = (cx_px - cx_K) * depth / fx
    Y = (cy_px - cy_K) * depth / fy
    Z = depth
    center = np.array([X, Y, Z])

    ca, sa = math.cos(angle), math.sin(angle)
    uW = np.array([ ca,  sa, 0.0])   # long edge, in the image plane
    uL = np.array([-sa,  ca, 0.0])   # short edge, perpendicular to it
    uH = np.array([0.0, 0.0, 1.0])   # parcel height ~ along the optical axis
                                     # for a ceiling-mounted, near-nadir camera

    hw, hl, hh = W3d/2, L3d/2, H3d/2
    signs = [(-1,-1,-1),( 1,-1,-1),( 1, 1,-1),(-1, 1,-1),
             (-1,-1, 1),( 1,-1, 1),( 1, 1, 1),(-1, 1, 1)]
    return np.array([center + sw*hw*uW + sl*hl*uL + sh*hh*uH
                     for sw, sl, sh in signs])


def draw_rotated_rect(ax, cx, cy, w, h, angle_rad, **kw):
    """Draw an oriented rectangle on a matplotlib axis."""
    ca, sa  = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh  = w / 2, h / 2
    corners = np.array([[-hw, -hh], [ hw, -hh], [ hw,  hh], [-hw,  hh]])
    R     = np.array([[ca, -sa], [sa, ca]])
    rotc  = corners @ R.T + np.array([cx, cy])
    ax.add_patch(plt.Polygon(rotc, fill=False, **kw))
    ax.annotate('', xy=(cx + ca*hw*0.6, cy + sa*hw*0.6), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color=kw.get('edgecolor','lime'), lw=1.5))


# ---------------------------------------------------------------------------
# Main predict
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, image_path, conf_thresh=0.5, intrinsics=None,
            out_path="prediction.png", nms_iou=0.3, device="cpu",
            max_labels=8):

    S = int(model.hparams.img_size)
    fx_ref = float(model.hparams.fx_ref)

    img    = Image.open(image_path).convert("L")
    W0, H0 = img.size
    tfm    = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize([0.5],[0.5])])
    x      = tfm(img).unsqueeze(0).to(device)

    # ---- intrinsics, rescaled to model input resolution ----
    if intrinsics is not None:
        fx_o, fy_o, cx_o, cy_o = intrinsics
        sx, sy = S / W0, S / H0
        fx_m, fy_m, cx_m, cy_m = fx_o*sx, fy_o*sy, cx_o*sx, cy_o*sy
    else:
        # assume the checkpoint's reference camera; metric output is only
        # correct if the real lens matches it
        fx_m = fy_m = fx_ref
        cx_m = cy_m = S / 2.0
        print(f"  [warn] no --intrinsics given; assuming fx=fy={fx_ref:.0f} at "
              f"{S}x{S}. Metric depth/W/L/H scale linearly with the true fx, "
              f"so pass --intrinsics for trustworthy numbers.")

    dets = detect(model, x, conf_thresh, nms_iou)
    if dets is None:
        print("No parcels detected above threshold.")
        return None

    N = len(dets['scores'])
    angles = angles_from(dets)
    m3 = decode_3d(dets, fx_m, fy_m, fx_ref)
    depth, W3d, L3d, H3d = m3['depth'], m3['W'], m3['L'], m3['H']

    print(f"\nDetected {N} parcel(s):")
    for i in range(N):
        print(f"  [{i}] score={dets['scores'][i]:.3f}"
              f"  cx={dets['cx'][i]:.1f}px  cy={dets['cy'][i]:.1f}px"
              f"  angle={math.degrees(angles[i]):.1f}deg"
              f"  depth={depth[i]:.3f}m"
              f"  W={W3d[i]:.3f}m  L={L3d[i]:.3f}m  H={H3d[i]:.3f}m")

    # ---- visualise ----
    fig = plt.figure(figsize=(14, 6))

    # Per-detection W/L/H text is only legible for a handful of boxes; these
    # scenes routinely hold 30 packages, where it collapses into an unreadable
    # pile. Past `max_labels`, fall back to indices that cross-reference the
    # table printed above.
    annotate = N <= max_labels

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.array(img), cmap="gray")
    sx_v, sy_v = W0/S, H0/S
    for i in range(N):
        draw_rotated_rect(ax1,
            dets['cx'][i]*sx_v, dets['cy'][i]*sy_v,
            dets['w'][i]*sx_v,  dets['h'][i]*sy_v,
            angles[i], edgecolor='lime', linewidth=2)
        if not annotate:
            ax1.text(dets['cx'][i]*sx_v, dets['cy'][i]*sy_v, str(i),
                     color='yellow', fontsize=7, ha='center', va='center',
                     fontweight='bold')
    ax1.set_title(f"2D rotated boxes ({N})")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 0.9, 10))
    all_pts = []
    for i in range(N):
        corners = build_corners_3d(
            dets['cx'][i], dets['cy'][i], depth[i],
            W3d[i], L3d[i], H3d[i], angles[i], fx_m, fy_m, cx_m, cy_m)
        all_pts.append(corners)
        col = colors[i % len(colors)]
        ax2.add_collection3d(Poly3DCollection(
            [[corners[j] for j in face] for face in FACES],
            alpha=0.15, edgecolor=col, facecolor=col))

        def mid(a, b): return (corners[a] + corners[b]) / 2
        if annotate:
            ax2.text(*mid(0,1), f"W={W3d[i]*100:.1f}cm", color='black', fontsize=9, ha='center', va='bottom')
            ax2.text(*mid(1,2), f"L={L3d[i]*100:.1f}cm", color='black', fontsize=9, ha='center', va='bottom')
            ax2.text(*mid(0,4), f"H={H3d[i]*100:.1f}cm", color='black', fontsize=9, ha='center', va='bottom')
        else:
            ax2.text(*corners.mean(0), str(i), color='black', fontsize=7,
                     ha='center', va='center', fontweight='bold')

    # Poly3DCollection does not drive autoscaling, and unequal axis scales make
    # a cube render as a slab — set both from the actual corner extents.
    if all_pts:
        pts = np.concatenate(all_pts, 0)
        lo, hi = pts.min(0), pts.max(0)
        pad = 0.05 * np.maximum(hi - lo, 1e-3)
        lo, hi = lo - pad, hi + pad
        ax2.set_xlim(lo[0], hi[0]); ax2.set_ylim(lo[1], hi[1]); ax2.set_zlim(lo[2], hi[2])
        ax2.set_box_aspect(tuple(hi - lo))

    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z depth (m)")
    ax2.set_title("3D bounding boxes"
                  + ("" if annotate else f" — indices match the table ({N})"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved -> {out_path}")
    return dets


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_model(ckpt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RTMDetLitModule.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    return model, device


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a single input image.")
    group.add_argument("--folder", help="Directory of input images; runs "
                                        "prediction on every image in it.")
    p.add_argument("--conf",  type=float, default=0.5)
    p.add_argument("--nms_iou", type=float, default=0.3,
                   help="rotated-IoU above which two detections are duplicates")
    p.add_argument("--max_labels", type=int, default=8,
                   help="annotate each 3D box with W/L/H only when at most this "
                        "many are detected; above it, boxes get indices that "
                        "match the printed table")
    p.add_argument("--out",     default="prediction.png",
                   help="Output overlay PNG path (--image mode).")
    p.add_argument("--out_dir", default="predictions",
                   help="Output directory for overlay PNGs (--folder mode).")
    p.add_argument("--intrinsics", type=float, nargs=4,
                   metavar=("FX","FY","CX","CY"),
                   help="Camera intrinsics of the input image(s), in ORIGINAL "
                        "image pixels. Required for correct metric output.")
    args = p.parse_args()
    intrinsics = tuple(args.intrinsics) if args.intrinsics else None

    model, device = load_model(args.ckpt)

    if args.image:
        predict(model, args.image, args.conf, intrinsics=intrinsics,
                out_path=args.out, nms_iou=args.nms_iou, device=device,
                max_labels=args.max_labels)
    else:
        folder = Path(args.folder)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS)
        if not images:
            print(f"No images found in {folder}")
        for img_path in images:
            print(f"\n=== {img_path.name} ===")
            predict(model, str(img_path), args.conf, intrinsics=intrinsics,
                    out_path=str(out_dir / f"{img_path.stem}_pred.png"),
                    nms_iou=args.nms_iou, device=device,
                    max_labels=args.max_labels)
