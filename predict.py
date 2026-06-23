import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from losses import Parcel3DLitModule


CORNER_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

FACES = [
    [0,1,2,3], [4,5,6,7],
    [0,1,5,4], [2,3,7,6],
    [1,2,6,5], [0,3,7,4],
]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation (Zhou et al. 2019) to 3x3 rotation matrix.
    rot6d: (..., 6)
    returns: (..., 3, 3)
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)


def build_corners_numpy(centers: np.ndarray,
                        sizes: np.ndarray,
                        rot6d: np.ndarray) -> np.ndarray:
    """
    Build 8 corners of each 3D box in camera frame.

    centers : (N, 3)
    sizes   : (N, 3)  w, h, d
    rot6d   : (N, 6)
    returns : (N, 8, 3)
    """
    N = centers.shape[0]
    R = rot6d_to_matrix(rot6d)  # (N, 3, 3)

    # unit cube corners: ±0.5 along each axis
    offsets = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ], dtype=np.float32)  # (8, 3)

    corners = np.zeros((N, 8, 3), dtype=np.float32)
    for i in range(N):
        # scale unit cube by box size
        scaled = offsets * sizes[i]          # (8, 3)
        # rotate
        rotated = scaled @ R[i].T            # (8, 3)
        # translate
        corners[i] = rotated + centers[i]   # (8, 3)

    return corners


def unproject(uv_norm: np.ndarray,
              depth: np.ndarray,
              K: np.ndarray,
              img_w: int = 640,
              img_h: int = 640) -> np.ndarray:
    """
    Lift normalised pixel coordinates + depth to 3D camera-frame points.

    uv_norm : (N, 2)  values in [0, 1]
    depth   : (N,)    metric depth (z-forward)
    K       : (3, 3)  intrinsics for the 640×640 model space
    returns : (N, 3)
    """
    # convert normalised coords to pixel coords
    px = uv_norm[:, 0] * img_w   # u
    py = uv_norm[:, 1] * img_h   # v

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (px - cx) / fx * depth
    y = (py - cy) / fy * depth
    z = depth

    return np.stack([x, y, z], axis=-1)  # (N, 3)


def project_corners(corners: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D corners (z-forward camera frame) to 2D image pixels.

    corners : (N, 8, 3)
    K       : (3, 3)
    returns : (N, 8, 2)
    """
    pts = corners.copy()
    z = np.clip(pts[..., 2:3], 1e-3, None)
    uv = pts[..., :2] / z
    uv = uv @ K[:2, :2].T + K[:2, 2]
    return uv  # (N, 8, 2)


@torch.no_grad()
def predict(ckpt: str,
            image_path: str,
            conf_thresh: float = 0.5,
            K: np.ndarray | None = None,
            out_path: str = "prediction.png",
            intrinsics: tuple | None = None):

    # ------------------------------------------------------------------ model
    model = Parcel3DLitModule.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()

    # ------------------------------------------------------------------ image
    img = Image.open(image_path).convert("L")
    W0, H0 = img.size

    tfm = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    x = tfm(img).unsqueeze(0)  # (1, 1, 640, 640)

    # ---------------------------------------------------------------- forward
    out = model(x)

    # out keys: pred_logits (1,Q,2), pred_uv (1,Q,2), pred_depth (1,Q,1),
    #           pred_size (1,Q,3),   pred_rot6d (1,Q,6)
    probs  = out["pred_logits"].softmax(-1)[0]   # (Q, 2)
    scores = probs[:, 1]                          # foreground score
    keep   = scores > conf_thresh

    if keep.sum() == 0:
        print("No parcels detected above confidence threshold.")
        return

    uv_norm = out["pred_uv"][0][keep].cpu().numpy()       # (N, 2)
    depth   = out["pred_depth"][0][keep].squeeze(-1).cpu().numpy()  # (N,)
    sizes   = out["pred_size"][0][keep].cpu().numpy()     # (N, 3)
    rot6d   = out["pred_rot6d"][0][keep].cpu().numpy()    # (N, 6)
    scores  = scores[keep].cpu().numpy()                  # (N,)

    # ----------------------------------------- intrinsics (model space = 640×640)
    if K is None:
        if intrinsics is not None:
            # intrinsics provided as (fx, fy, cx, cy) for the original image;
            # rescale to 640×640 model space
            fx, fy, cx, cy = intrinsics
            sx, sy = 640.0 / W0, 640.0 / H0
            K = np.array(
                [[fx * sx,      0, cx * sx],
                 [     0,  fy * sy, cy * sy],
                 [     0,       0,       1]], dtype=np.float32
            )
        else:
            K = np.array(
                [[1100,   0, 320],
                 [   0, 1100, 320],
                 [   0,    0,   1]], dtype=np.float32
            )

    # lift UV + depth → 3D centers in camera frame (z-forward)
    centers = unproject(uv_norm, depth, K, img_w=640, img_h=640)  # (N, 3)

    # build 8-corner boxes
    corners = build_corners_numpy(centers, sizes, rot6d)  # (N, 8, 3)

    # ----------------------------------------------------------------- report
    print(f"Detected {len(corners)} parcel(s)")
    for i in range(len(corners)):
        w, l, h = sizes[i]
        print(
            f"  [{i}] score={scores[i]:.3f}"
            f"  center={[round(v,4) for v in centers[i].tolist()]}"
            f"  W={w:.4f}m  L={l:.4f}m  H={h:.4f}m"
            f"  depth={depth[i]:.4f}m"
        )

    # --------------------------------------------------------------- visualise
    fig = plt.figure(figsize=(14, 6))

    # -- 2D projection over original image --
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.array(img), cmap="gray")

    # rescale intrinsics from 640×640 model space → original image resolution
    sx, sy = W0 / 640.0, H0 / 640.0
    K_img = K.copy()
    K_img[0] *= sx
    K_img[1] *= sy

    # project corners using original-resolution intrinsics
    uv_proj = project_corners(corners, K_img)  # (N, 8, 2)

    for box in uv_proj:
        for a, b in CORNER_EDGES:
            ax1.plot(
                [box[a, 0], box[b, 0]],
                [box[a, 1], box[b, 1]],
                color="lime", linewidth=2,
            )
    ax1.set_title("2D projection")
    ax1.axis("off")

    # -- 3D view --
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for box in corners:
        ax2.scatter(box[:, 0], box[:, 1], box[:, 2], s=10)
        polys = [[box[i] for i in f] for f in FACES]
        ax2.add_collection3d(
            Poly3DCollection(polys, alpha=0.2, edgecolor="k")
        )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("3D bounding boxes (camera frame, z-forward)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  required=True,  help="Lightning checkpoint (.ckpt)")
    p.add_argument("--image", required=True,  help="Input image path")
    p.add_argument("--conf",  type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--out",   type=str,   default="prediction.png")
    p.add_argument(
        "--intrinsics", type=float, nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help="Camera intrinsics for the input image (fx fy cx cy). "
             "Enables metric W/L/H estimation. If omitted, uses training defaults.",
    )
    args = p.parse_args()

    predict(args.ckpt, args.image, args.conf, out_path=args.out,
            intrinsics=tuple(args.intrinsics) if args.intrinsics else None)