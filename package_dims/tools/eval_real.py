"""Run a trained custom_3d_model checkpoint on the real frames of cam_000CDF0AD90F: per package the oriented bounding
box (image) and the 3D box, with length x width x height in cm, next to the Readme ground truth.

    python3 tools/eval_real.py --ckpt checkpoints/cam5_v4/<best>.ckpt        # -> predictions/real_eval/

L = long horizontal side, W = short horizontal side, H = height from the floor (pallet included), as in the Readme.
The package of a frame is the largest detection containing its DMC label (else the nearest one); frames without a
DMC use the highest-scoring detection. Outputs in --out_dir:
    results.csv    one row per frame: predicted / true L W H (cm), errors, score, rotated box corners (px)
    results.json   every detection: score, rotated box (centre, size, angle, 4 corners px), 3D box corners (camera
                   frame, m), depth, L W H cm
    <frame>.jpg    overlay: all detections (thin), the package (thick) with predicted and true sizes
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/daryna/code_ws/study/3D_detection/custom_3d_model")
sys.path.insert(0, MODEL_DIR)
from model import decode_3d  # noqa: E402
from predict import angles_from, build_corners_3d, detect, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def rbox_corners(cx, cy, w, h, a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[cx + dx * ca - dy * sa, cy + dx * sa + dy * ca]
                     for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))])


@torch.no_grad()
def run_frame(model, device, png, K, conf, nms_iou):
    """-> list of detections (original image pixels / metres / cm)."""
    S = int(model.hparams.img_size)
    img = Image.open(png).convert("L")
    W0, H0 = img.size
    x = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize([0.5], [0.5])])(img).unsqueeze(0).to(device)
    sx, sy = S / W0, S / H0
    fx_m, fy_m = K["fx"] * sx, K["fy"] * sy
    dets = detect(model, x, conf, nms_iou)
    if dets is None:
        return []
    ang = angles_from(dets)
    m3 = decode_3d(dets, fx_m, fy_m, float(model.hparams.fx_ref))
    out = []
    for i in range(len(dets["scores"])):
        cx, cy, w, h = dets["cx"][i] / sx, dets["cy"][i] / sy, dets["w"][i] / sx, dets["h"][i] / sy
        a = float(ang[i])
        depth, Lm, Wm, Hm = (float(m3[k][i]) for k in ("depth", "W", "L", "H"))  # model W = long side
        box3d = build_corners_3d(cx, cy, depth, Lm, Wm, Hm, a, K["fx"], K["fy"], K["cx"], K["cy"])
        out.append({"score": float(dets["scores"][i]), "cx_px": float(cx), "cy_px": float(cy), "w_px": float(w),
                    "h_px": float(h), "angle_deg": math.degrees(a),
                    "box_2d_px": np.round(rbox_corners(cx, cy, w, h, a), 1).tolist(),
                    "box_3d_cam_m": np.round(box3d, 4).tolist(), "depth_m": depth,
                    "L_cm": 100 * Lm, "W_cm": 100 * Wm, "H_cm": 100 * Hm})
    return out


def pick(dets, gt):
    if not dets:
        return None
    dmc = gt.get("dmc_corners_px")
    if not dmc:
        return max(range(len(dets)), key=lambda i: dets[i]["score"])
    c = np.mean(dmc, 0)
    inside = [i for i, d in enumerate(dets)
              if cv2.pointPolygonTest(np.float32(d["box_2d_px"]), (float(c[0]), float(c[1])), False) >= 0]
    if inside:  # the package is the largest box around its label (small boxes on the label itself happen)
        return max(inside, key=lambda i: dets[i]["w_px"] * dets[i]["h_px"])
    return min(range(len(dets)), key=lambda i: math.hypot(dets[i]["cx_px"] - c[0], dets[i]["cy_px"] - c[1]))


def draw(png, dets, k, gt, path):
    im = cv2.cvtColor(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)[:1024], cv2.COLOR_GRAY2BGR)
    for i, d in enumerate(dets):
        cv2.polylines(im, [np.int32(d["box_2d_px"])], True, (0, 255, 0) if i == k else (0, 160, 255),
                      3 if i == k else 1, cv2.LINE_AA)
    lines = [f"true  L {100 * gt['L_m']:.0f}  W {100 * gt['W_m']:.0f}  H {100 * gt['H_m']:.0f} cm"]
    if k is not None:
        d = dets[k]
        lines.insert(0, f"pred  L {d['L_cm']:.0f}  W {d['W_cm']:.0f}  H {d['H_cm']:.0f} cm  (score {d['score']:.2f})")
    else:
        lines.insert(0, "pred  no detection")
    for j, t in enumerate(lines):
        cv2.putText(im, t, (12, 34 + 32 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(im, t, (12, 34 + 32 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "datasets" / "REAL_CAM_000CDF0AD90F"))
    ap.add_argument("--out_dir", default=str(ROOT / "predictions" / "real_eval"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms_iou", type=float, default=0.3)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, device = load_model(args.ckpt)
    rows, everything, err = [], {}, []
    for js in sorted(Path(args.data).glob("shard_*/*.json")):
        meta = json.loads(js.read_text())
        gt, K = meta["ground_truth"], meta["camera_intrinsics"]
        png = js.with_name(js.stem + ".gray.png")
        dets = run_frame(model, device, png, K, args.conf, args.nms_iou)
        k = pick(dets, gt)
        everything[js.stem] = {"ground_truth": gt, "package_index": k, "detections": dets}
        draw(png, dets, k, gt, out / f"{js.stem}.jpg")
        row = {"frame": js.stem, "package": gt["package"], "n_det": len(dets),
               "true_L_cm": 100 * gt["L_m"], "true_W_cm": 100 * gt["W_m"], "true_H_cm": 100 * gt["H_m"]}
        if k is not None:
            d = dets[k]
            row.update(score=round(d["score"], 3), pred_L_cm=round(d["L_cm"], 1), pred_W_cm=round(d["W_cm"], 1),
                       pred_H_cm=round(d["H_cm"], 1), box_2d_px=d["box_2d_px"])
            e = [d["L_cm"] - row["true_L_cm"], d["W_cm"] - row["true_W_cm"], d["H_cm"] - row["true_H_cm"]]
            row.update(err_L_cm=round(e[0], 1), err_W_cm=round(e[1], 1), err_H_cm=round(e[2], 1))
            err.append(e)
        rows.append(row)
    keys = ["frame", "package", "n_det", "score", "pred_L_cm", "pred_W_cm", "pred_H_cm", "true_L_cm", "true_W_cm",
            "true_H_cm", "err_L_cm", "err_W_cm", "err_H_cm", "box_2d_px"]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    (out / "results.json").write_text(json.dumps(everything, indent=1))
    print(f"{'frame':9s} {'pkg':>3s} {'score':>5s}   {'pred L x W x H cm':>20s}   {'true L x W x H cm':>20s}   err L/W/H cm")
    for r in rows:
        if "pred_L_cm" in r:
            print(f"{r['frame']:9s} {r['package']:3d} {r['score']:5.2f}   "
                  f"{r['pred_L_cm']:6.0f} {r['pred_W_cm']:6.0f} {r['pred_H_cm']:6.0f}   "
                  f"{r['true_L_cm']:6.0f} {r['true_W_cm']:6.0f} {r['true_H_cm']:6.0f}   "
                  f"{r['err_L_cm']:+6.0f} {r['err_W_cm']:+6.0f} {r['err_H_cm']:+6.0f}")
        else:
            print(f"{r['frame']:9s} {r['package']:3d}   no detection")
    if err:
        e = np.abs(np.array(err))
        print(f"\ndetected {len(err)}/{len(rows)} frames; MAE  L {e[:, 0].mean():.1f}  W {e[:, 1].mean():.1f}  "
              f"H {e[:, 2].mean():.1f} cm;  median  L {np.median(e[:, 0]):.1f}  W {np.median(e[:, 1]):.1f}  "
              f"H {np.median(e[:, 2]):.1f} cm")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
