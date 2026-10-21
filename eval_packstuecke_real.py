"""
Score a checkpoint against the REAL `packstuecke/` frames and their hand-drawn
rotated boxes, and draw the overlays.

This is the only genuinely out-of-domain test in the repo: the model never saw
a real frame, only the `TASLER_PACK/` imitation of one. Two things are measured
separately, because only one of them can be checked honestly:

  * **2D detection** — precision / recall / F1 / mean IoU against the human
    labels in `packstuecke/*.txt` (`class cx cy w h angle_deg`, normalised;
    verified to agree with the `robndbox` entries in the matching `*.xml`).
    Rotated IoU needs no camera calibration, so these numbers stand on their
    own.

  * **Metric 3D** — W/L/H and depth. The real frames ship no metric ground
    truth, so nothing here is a measurement; the script reports the predicted
    distributions and a plausibility check against standard pallet footprints.
    W/L/H come straight from the metric dimension head and do NOT depend on
    the camera. Depth does, and the frames carry no per-image calibration, so
    `--fx_native` is an ASSUMPTION (default 3250, the JAI GODA-5000M / KOWA
    LM16HC that covers most of Dachser's fleet) and depth scales linearly with
    it.

Framing: the real frames come in two shapes - 1280x1280 with black bands on the
right and bottom, and the bare ~1200x1000 crop. Both are put on a square canvas
of side max(W, H) with the image at the top-left, which is exactly the padding
the 1280 ones already have, before the resize to the model's 640. That keeps the
scale isotropic, so a rotated GT box stays a rotated box and the model sees the
framing it was trained on.

Usage:
    python3 eval_packstuecke_real.py --ckpt checkpoints_multi/rtmdet_pack/last.ckpt
    python3 eval_packstuecke_real.py --ckpt ... --limit 400 --viz_n 24
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from geometry import rotated_iou, rbox_corners
from losses import RTMDetLitModule
from model import decode_3d
from predict import decode_level, nms_rotated, angles_from, STRIDES, DET_KEYS

HERE = Path(__file__).resolve().parent


def load_gt(txt_path, W, H):
    """packstuecke label file -> [(cx, cy, w, h, angle_rad)] in image pixels."""
    out = []
    for line in txt_path.read_text().splitlines():
        p = line.split()
        if len(p) != 6:
            continue
        try:
            _, cx, cy, w, h, ang = (float(v) for v in p)
        except ValueError:
            continue                     # classes.txt and friends
        out.append((cx * W, cy * H, w * W, h * H, math.radians(ang)))
    return out


def square_pad(img):
    """Put the frame top-left on a square black canvas, as the real 1280x1280
    variants already are. Isotropic, so labels only need the same scale."""
    W, H = img.size
    S = max(W, H)
    if W == H:
        return img, S
    canvas = Image.new("L", (S, S), 0)
    canvas.paste(img, (0, 0))
    return canvas, S


def detect_all(model, x, conf, nms_iou):
    level_outs = model.model(x)
    all_dets = {k: [] for k in DET_KEYS}
    for out, stride in zip(level_outs, STRIDES):
        d = decode_level(out, stride, conf)
        if d is not None:
            for k in all_dets:
                all_dets[k].append(d[k])
    if not all_dets['scores']:
        return None
    return nms_rotated({k: np.concatenate(v) for k, v in all_dets.items()},
                       nms_iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    # test_images/ holds the real frames never used for training (see its
    # SPLIT_MANIFEST.json). packstuecke/ is now the real TRAINING set: pointing
    # this at it would score a model on frames it was trained on.
    ap.add_argument("--data", default=str(HERE / "test_images"))
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--nms_iou", type=float, default=0.3)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--viz_n", type=int, default=24)
    ap.add_argument("--viz_out", default="/tmp/pack_real_viz")
    ap.add_argument("--fx_native", type=float, default=3250.0,
                    help="assumed focal length in the frame's own pixels; "
                         "depth scales linearly with it, W/L/H do not")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RTMDetLitModule.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    S = int(model.hparams.img_size)
    fx_ref = float(model.hparams.fx_ref)

    data = Path(args.data)
    pairs = []
    for txt in sorted(data.glob("*.txt")):
        if txt.stem == "classes":
            continue
        jpg = txt.with_suffix(".jpg")
        if jpg.exists():
            pairs.append((jpg, txt))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"{len(pairs)} labelled real frames in {data}")

    viz_dir = Path(args.viz_out)
    viz_dir.mkdir(parents=True, exist_ok=True)

    tfm = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize([0.5], [0.5])])

    n_tp = n_fp = n_fn = 0
    ious, angs, metrics = [], [], []
    viz_rows = []

    for k, (jpg, txt) in enumerate(pairs):
        img0 = Image.open(jpg).convert("L")
        W0, H0 = img0.size
        gts = load_gt(txt, W0, H0)
        if not gts:
            continue

        sq, side = square_pad(img0)
        s = S / side                            # isotropic
        gt_m = [(cx * s, cy * s, w * s, h * s, a) for cx, cy, w, h, a in gts]

        with torch.no_grad():
            x = tfm(sq).unsqueeze(0).to(device)
            dets = detect_all(model, x, args.conf, args.nms_iou)

        used = [False] * len(gt_m)
        pred_rows = []
        if dets is not None:
            angles = angles_from(dets)
            # intrinsics of the SQUARE canvas at model resolution (see module
            # docstring: this is an assumption, not a calibration)
            fx_m = fy_m = args.fx_native * s
            m3 = decode_3d(dets, fx_m, fy_m, fx_ref)
            depth, W3d, L3d, H3d = m3['depth'], m3['W'], m3['L'], m3['H']
            p_unit = m3['p_unit']

            for i in np.argsort(-dets['scores']):
                pr = (float(dets['cx'][i]), float(dets['cy'][i]),
                      float(dets['w'][i]), float(dets['h'][i]), float(angles[i]))
                best, bj = 0.0, -1
                for j in range(len(gt_m)):
                    if used[j]:
                        continue
                    v = rotated_iou(pr, gt_m[j])
                    if v > best:
                        best, bj = v, j
                hit = bj >= 0 and best >= args.iou_thresh
                if hit:
                    used[bj] = True
                    n_tp += 1
                    ious.append(best)
                    g = gt_m[bj]
                    gl, gs = max(g[2], g[3]), min(g[2], g[3])
                    ga = g[4] % math.pi
                    if gs > 1e-6 and gl / gs >= 1.15:
                        # GT w/h are not long-edge sorted; put its angle on the
                        # long edge before comparing with the head's convention
                        if g[3] > g[2]:
                            ga = (ga + math.pi / 2) % math.pi
                        d = abs(pr[4] - ga) % math.pi
                        angs.append(math.degrees(min(d, math.pi - d)))
                    metrics.append((float(depth[i]), float(W3d[i]),
                                    float(L3d[i]), float(H3d[i]),
                                    float(p_unit[i])))
                else:
                    n_fp += 1
                pred_rows.append((pr, float(dets['scores'][i]), hit))
        n_fn += used.count(False)

        if len(viz_rows) < args.viz_n and k % max(1, len(pairs) // max(args.viz_n, 1)) == 0:
            viz_rows.append((jpg.name, sq.resize((S, S)), gt_m, pred_rows))

    # ---- report --------------------------------------------------------
    prec = n_tp / max(n_tp + n_fp, 1)
    rec = n_tp / max(n_tp + n_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\nREAL packstuecke, 2D detection "
          f"(conf>={args.conf}, IoU>={args.iou_thresh}, nms={args.nms_iou})")
    print(f"  TP={n_tp}  FP={n_fp}  FN={n_fn}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    if ious:
        print(f"  mean IoU (TP)   = {np.mean(ious):.3f}")
    if angs:
        print(f"  angle MAE elong = {np.mean(angs):.2f} deg  ({len(angs)} TPs)")
    if metrics:
        m = np.array(metrics)
        print(f"\n  metric output on {len(m)} matched units. There is NO metric "
              f"ground truth for these frames, so none of this is a measurement.")
        print(f"  W/L/H do not depend on fx. depth does, and assumes "
              f"fx={args.fx_native:.0f} px.")
        for i, n in enumerate(["depth", "W", "L", "H"]):
            print(f"    {n:6s} median {np.median(m[:, i]):6.3f} m   "
                  f"p5 {np.percentile(m[:, i], 5):6.3f}   "
                  f"p95 {np.percentile(m[:, i], 95):6.3f}")
        print(f"  predicted type = palletised unit: {np.mean(m[:, 4] >= 0.5):.1%}")
        # Plausibility, not accuracy: nearly every labelled unit here sits on a
        # EUR (1.2x0.8) or industrial (1.2x1.0) pallet or a half pallet
        # (0.8x0.6), so a long edge outside ~0.7-1.5 m is at least suspicious.
        band = (m[:, 1] >= 0.7) & (m[:, 1] <= 1.5)
        print(f"  long edge W in the 0.7-1.5 m pallet band: {band.mean():.1%}   "
              f"(plausibility check only)")

    # ---- overlays ------------------------------------------------------
    if viz_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        n = len(viz_rows)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.5),
                                 facecolor="#1a1a19")
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, (name, im, gt_m, preds) in zip(np.atleast_1d(axes).ravel(), viz_rows):
            ax.imshow(np.array(im), cmap="gray", vmin=0, vmax=255)
            for g in gt_m:
                ax.add_patch(Polygon(rbox_corners(*g), closed=True, fill=False,
                                     edgecolor="#1baf7a", linewidth=2.0))
            for pr, sc, hit in preds:
                ax.add_patch(Polygon(rbox_corners(*pr), closed=True, fill=False,
                                     edgecolor="#2a78d6" if hit else "#e34948",
                                     linewidth=2.0, linestyle="-"))
                ax.text(pr[0], pr[1], f"{sc:.2f}", color="#ffffff", fontsize=7,
                        ha="center", va="center")
            ax.set_title(name, color="#c3c2b7", fontsize=8)
        fig.suptitle("real packstuecke — ground truth (green) vs prediction "
                     "(blue = matched, red = false positive)",
                     color="#ffffff", fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = viz_dir / "packstuecke_predictions.png"
        fig.savefig(out, facecolor="#1a1a19", dpi=130)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
