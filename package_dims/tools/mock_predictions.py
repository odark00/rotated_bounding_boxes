"""Illustration of what the predictions will look like: on raw frames of cam_000CDF0AD90F, a 3D box per package with
length / width / height in cm. NOT model output: the box is the Readme size fitted to the package by hand-marked
outlines, then perturbed by 2-9 cm per dimension (random sign) to mimic a good prediction.

    python3 tools/mock_predictions.py            # -> predictions/mockup/mock_pred_*.jpg + overview.jpg

The outline of each package (undistorted full-res pixels, top face + visible sides) was read off the frames; the
floor position / yaw of the 3D box (Readme L x W x H) is fitted to it by silhouette IoU. Drawn on the raw
(distorted) image: every box edge is sampled and passed through the lens distortion.
"""
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from camera import Camera  # noqa: E402

CAM_DIR = os.path.join(os.path.dirname(ROOT), "cam_000CDF0AD90F")
REAL = os.path.join(ROOT, "datasets", "REAL_CAM_000CDF0AD90F", "shard_00000000")
OUT = os.path.join(ROOT, "predictions", "mockup")
# frame -> (outline (undistorted full-res px), initial yaw of the L side in degrees)
OUTLINES = {
    0: ([(900, 680), (1520, 950), (1308, 1347), (733, 1050)], 23.5),
    2: ([(275, 1175), (875, 1192), (783, 1892), (258, 1833)], 95.0),
    4: ([(800, 658), (1375, 675), (1367, 1500), (783, 1483)], 91.0),
    6: ([(558, 892), (1167, 1042), (892, 1833), (350, 1683)], 107.0),
    8: ([(1450, 808), (1758, 1375), (858, 1850), (567, 1300)], 151.5),
    9: ([(1864, 509), (2424, 1309), (1970, 1691), (1309, 879)], 55.0),
    10: ([(358, 897), (933, 897), (958, 1320), (342, 1320), (303, 1033)], -1.0),
    14: ([(1653, 697), (1780, 892), (1758, 1083), (1300, 1383), (1050, 1358), (930, 1170)], -33.0),
    16: ([(1025, 942), (1925, 942), (1925, 1492), (1025, 1492)], 0.0),
    17: ([(195, 985), (520, 962), (620, 962), (640, 1330), (550, 1368), (235, 1355)], 85.0),
}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
DS = 4


def box_corners(cam, X, Y, yaw, L, W, H):
    """Camera-frame corners: bottom 0-3 on the floor, top 4-7; side 0-1 = L (along yaw), 1-2 = W."""
    c, s = math.cos(yaw), math.sin(yaw)
    xy = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]]) @ np.array([[c, s], [-s, c]])
    xy += (X, Y)
    return np.concatenate([np.c_[xy, np.full(4, cam.mount_h)], np.c_[xy, np.full(4, cam.mount_h - H)]])


def fit(cam, outline, yaw0, L, W, H, shape):
    target = np.zeros((shape[0] // DS, shape[1] // DS), np.uint8)
    cv2.fillPoly(target, [np.round(np.array(outline) / DS * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    target = target.astype(bool)

    def score(p):
        m = np.zeros_like(target, np.uint8)
        uv = cam.project(box_corners(cam, *p, L, W, H)) / DS
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv * 16).astype(np.int32)), 1, cv2.LINE_8, 4)
        m = m.astype(bool)
        return (m & target).sum() / max(1, (m | target).sum())
    X, Y = cam.backproject(np.mean(outline, 0)[None], cam.mount_h - H / 2)[0, :2]
    p, sc = np.array([X, Y, math.radians(yaw0)]), 0.0
    sc = score(p)
    for st_xy, st_a in ((0.08, 0.05), (0.04, 0.025), (0.02, 0.012), (0.01, 0.006), (0.005, 0.003), (0.0025, 0.0015)):
        improved = True
        while improved:
            improved = False
            for j, st in ((0, st_xy), (1, st_xy), (2, st_a)):
                for sg in (-1, 1):
                    q = p.copy()
                    q[j] += sg * st
                    s2 = score(q)
                    if s2 > sc:
                        p, sc, improved = q, s2, True
    return p, sc


def raw_px(cam, P):
    """Camera-frame points -> raw (distorted) pixels."""
    return cam.distort(cam.project(P))


def label(img, text, at, scale=1.5, color=(255, 255, 255), bg=(20, 20, 20), taken=None, push=(0.0, 0.0)):
    """Text on a dark box centred at `at`; moved along `push` (unit vector) until it is clear of `taken` boxes."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    for k in range(25):
        cx, cy = at[0] + push[0] * 30 * k, at[1] + push[1] * 30 * k
        x, y = int(cx - tw / 2), int(cy + th / 2)
        x, y = int(np.clip(x, 8, img.shape[1] - tw - 8)), int(np.clip(y, th + 8, img.shape[0] - base - 8))
        r = (x - 14, y - th - 14, x + tw + 14, y + base + 12)
        if not taken or all(r[2] < q[0] or q[2] < r[0] or r[3] < q[1] or q[3] < r[1] for q in taken):
            break
    if taken is not None:
        taken.append(r)
    ov = img.copy()
    cv2.rectangle(ov, (x - 10, y - th - 10), (x + tw + 10, y + base + 8), bg, -1)
    cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 2, cv2.LINE_AA)


def draw(cam, img, P, dims_cm):
    top_col, side_col, bot_col = (0, 215, 255), (0, 255, 120), (0, 180, 90)
    ov = img.copy()  # light fill on the top face
    top = raw_px(cam, np.concatenate([P[4 + i] + (P[4 + (i + 1) % 4] - P[4 + i]) * t[:, None]
                                      for i in range(4) for t in [np.linspace(0, 1, 30, endpoint=False)]]))
    cv2.fillPoly(ov, [np.round(top).astype(np.int32)], (0, 200, 255))
    cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
    for a, b in EDGES:
        pts = raw_px(cam, P[a] + (P[b] - P[a]) * np.linspace(0, 1, 40)[:, None])
        col, th = (top_col, 5) if a >= 4 else (bot_col, 3) if b < 4 else (side_col, 3)
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, th, cv2.LINE_AA)
    c_img = raw_px(cam, P[4:].mean(0)[None])[0]

    def out(a, b, d=70):
        """Label anchor: edge midpoint pushed away from the box centre; and that direction."""
        m = raw_px(cam, ((P[a] + P[b]) / 2)[None])[0]
        u = (m - c_img) / (np.linalg.norm(m - c_img) + 1e-9)
        return m + u * d, u
    L_cm, W_cm, H_cm = dims_cm
    taken = []
    for text, (at, u) in ((f"L {L_cm:.0f} cm", out(4, 5)), (f"W {W_cm:.0f} cm", out(5, 6))):
        label(img, text, at, taken=taken, push=u)
    v = max(range(4), key=lambda i: np.linalg.norm(np.diff(raw_px(cam, P[[i, i + 4]]), axis=0)))  # longest vertical
    at, u = out(v, v + 4, 90)
    label(img, f"H {H_cm:.0f} cm", at, taken=taken, push=u)


def main():
    os.makedirs(OUT, exist_ok=True)
    cam = Camera(CAM_DIR, 1.0)
    rng = np.random.default_rng(15)
    tiles, summary = [], []
    for f, (outline, yaw0) in OUTLINES.items():
        gt = json.load(open(os.path.join(REAL, f"{f:08d}.json")))["ground_truth"]
        L, W, H = gt["L_m"], gt["W_m"], gt["H_m"]
        raw = cv2.imread(os.path.join(CAM_DIR, gt["source"] + ".png"), cv2.IMREAD_GRAYSCALE)
        p, iou = fit(cam, outline, yaw0, L, W, H, raw.shape)
        err = rng.uniform(2, 9, 3) * rng.choice([-1, 1], 3)  # cm
        pred = np.array([L, W, H]) * 100 + err
        pp = p + (rng.normal(0, 0.015), rng.normal(0, 0.015), math.radians(rng.normal(0, 0.6)))
        P = box_corners(cam, *pp, *(pred / 100))
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        draw(cam, img, P, pred)
        lines = [f"Package {gt['package']}   predicted  L {pred[0]:.0f} x W {pred[1]:.0f} x H {pred[2]:.0f} cm",
                 f"measured (Readme)  L {L * 100:.0f} x W {W * 100:.0f} x H {H * 100:.0f} cm   "
                 f"error {err[0]:+.0f} / {err[1]:+.0f} / {err[2]:+.0f} cm"]
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (1700, 140), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
        for k, t in enumerate(lines):
            cv2.putText(img, t, (24, 55 + 55 * k), cv2.FONT_HERSHEY_DUPLEX, 1.3 if k == 0 else 1.1,
                        (255, 255, 255) if k == 0 else (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, "illustration: simulated prediction (true size +/- 2-9 cm), not model output",
                    (24, img.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (190, 190, 190), 2, cv2.LINE_AA)
        name = f"mock_pred_frame{f:02d}_pkg{gt['package']}.jpg"
        cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(cv2.resize(img, (800, 640), interpolation=cv2.INTER_AREA))
        summary.append((f, gt["package"], iou, pred, err))
        print(f"frame {f:2d} pkg {gt['package']}  fit IoU {iou:.2f}  pred {pred.round(0)}  err {err.round(1)} cm -> {name}")
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(OUT, "overview.jpg"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
