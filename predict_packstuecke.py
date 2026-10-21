"""
Per-image 2D + 3D prediction overlays for real packstuecke-style frames.

For each frame, writes one PNG with two panels:

  left   the frame with the model's rotated boxes; where a human label exists
         it is drawn too (green), and each prediction is blue if it matches a
         label at >= --iou_thresh or red if it does not. Frames without a label
         file get plain predictions (orange).
  right  the same detections as 3D boxes in the camera frame, with W x L x H,
         depth, and the predicted object type

plus `predictions.json` (every detection, every frame), `metrics.txt` (on the
labelled frames) and contact sheets of the labelled frames, 49 per page.

Two things differ from `predict.py --folder`, and both matter here:

  * **Square padding.** The real frames come 1280x1280 with black bands, or as
    the bare ~1200x1000 crop. Both are put top-left on a square canvas before
    the resize to the model's 640 - the framing the model was trained on.
    `predict.py` resizes straight to a square, which is anisotropic on the
    crops and turns a rotated box into a parallelogram on the way back.
  * **Ground truth is drawn** where `*.txt` labels exist.

**Reading the 3D panel.** W/L/H come from the metric dimension head and need no
camera calibration. Depth - and with it where the box sits along the optical
axis - needs `fx`, and these frames carry none, so `--fx_native` (default 3250,
the JAI GODA-5000M / KOWA LM16HC that covers most of the fleet) is an
assumption that depth scales linearly with. On real frames the sizes are
dominated by the learned class prior rather than measured from the image (see
eval_reports/rtmdet_pack_v3/SUMMARY.md): there is no metric ground truth for
these frames to learn or check them against.

Usage:
    python3 predict_packstuecke.py --ckpt <ckpt>                 # all of test_images/
    python3 predict_packstuecke.py --ckpt <ckpt> --n 50          # evenly spaced sample
    python3 predict_packstuecke.py --ckpt <ckpt> --labelled_only
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from eval_packstuecke_real import load_gt, square_pad, detect_all
from geometry import rotated_iou, rbox_corners
from losses import RTMDetLitModule
from model import decode_3d
from predict import FACES, angles_from, build_corners_3d

HERE = Path(__file__).resolve().parent

C_GT = "#1baf7a"      # human label
C_HIT = "#2a78d6"     # prediction that matches one
C_MISS = "#e34948"    # prediction that does not
C_PRED = "#eb6834"    # prediction on a frame with no label to compare against
BG = "#fcfcfb"
INK = "#0b0b0b"


def draw_rbox(ax, rb, color, lw=2.0, label=None):
    ax.add_patch(Polygon(rbox_corners(*rb), closed=True, fill=False,
                         edgecolor=color, linewidth=lw))
    # a stub along the long edge, so the predicted orientation is visible even
    # when the box is nearly square
    ca, sa = math.cos(rb[4]), math.sin(rb[4])
    ax.plot([rb[0], rb[0] + ca * rb[2] * 0.42],
            [rb[1], rb[1] + sa * rb[2] * 0.42], color=color, lw=lw)
    if label:
        ax.text(rb[0], rb[1], label, color="#ffffff", fontsize=8,
                ha="center", va="center", fontweight="bold",
                bbox=dict(facecolor="#000000", alpha=0.45, pad=1.2, lw=0))


def contact_sheets(sheet, out_dir, per_page=49, cols=7):
    pages = [sheet[i:i + per_page] for i in range(0, len(sheet), per_page)]
    for pno, page in enumerate(pages, 1):
        rows_n = (len(page) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.5, rows_n * 2.6),
                                 facecolor="#1a1a19")
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, (name, disp, gt_m, rws) in zip(np.atleast_1d(axes).ravel(), page):
            ax.imshow(disp, cmap="gray", vmin=0, vmax=255)
            for g in gt_m:
                ax.add_patch(Polygon(rbox_corners(*g), closed=True, fill=False,
                                     edgecolor=C_GT, linewidth=1.4))
            for r in rws:
                ax.add_patch(Polygon(rbox_corners(*r["rbox"]), closed=True,
                                     fill=False, linewidth=1.4,
                                     edgecolor=C_HIT if r["matched"] else C_MISS))
            ax.set_title(name, color="#c3c2b7", fontsize=7)
        fig.suptitle(f"held-out real frames, page {pno}/{len(pages)} — label (green) "
                     f"· matched prediction (blue) · false positive (red)",
                     color="#ffffff", fontsize=12, y=0.997)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(out_dir / f"_contact_sheet_{pno:02d}.png", dpi=110,
                    facecolor="#1a1a19")
        plt.close(fig)
    return len(pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    # test_images/ holds the real frames never used for training (see its
    # SPLIT_MANIFEST.json); packstuecke/ is the real TRAINING set.
    ap.add_argument("--data", default=str(HERE / "test_images"))
    ap.add_argument("--out_dir", default=None,
                    help="default: eval_reports/<run>/test_images_predictions")
    ap.add_argument("--n", type=int, default=0,
                    help="evenly spaced sample of this many frames (0 = all)")
    ap.add_argument("--labelled_only", action="store_true")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--nms_iou", type=float, default=0.3)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--fx_native", type=float, default=3250.0)
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    run = ckpt.parent.name.replace("_best", "")
    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "eval_reports" / run / "test_images_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RTMDetLitModule.load_from_checkpoint(str(ckpt), map_location=device)
    model.eval().to(device)
    S = int(model.hparams.img_size)
    fx_ref = float(model.hparams.fx_ref)
    epoch = torch.load(str(ckpt), map_location="cpu", weights_only=False).get("epoch")

    data = Path(args.data)
    frames = sorted(data.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))
    if args.labelled_only:
        frames = [f for f in frames if f.with_suffix(".txt").exists()]
    if args.n:
        step = max(1, len(frames) // args.n)
        frames = frames[::step][:args.n]
    n_lab = sum(f.with_suffix(".txt").exists() for f in frames)
    print(f"{len(frames)} frames from {data} ({n_lab} labelled, "
          f"{len(frames) - n_lab} unlabelled) -> {out_dir}")

    tfm = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize([0.5], [0.5])])
    sheet, records = [], {}
    n_tp = n_fp = n_fn = 0
    ious, sizes_tp = [], []

    for k, jpg in enumerate(frames):
        txt = jpg.with_suffix(".txt")
        labelled = txt.exists()
        img0 = Image.open(jpg).convert("L")
        gts = load_gt(txt, *img0.size) if labelled else []
        sq, side = square_pad(img0)
        s = S / side
        gt_m = [(cx * s, cy * s, w * s, h * s, a) for cx, cy, w, h, a in gts]
        disp = np.array(sq.resize((S, S)))

        with torch.no_grad():
            dets = detect_all(model, tfm(sq).unsqueeze(0).to(device),
                              args.conf, args.nms_iou)

        rows = []
        fx_m = fy_m = args.fx_native * s
        cx_m = cy_m = S / 2.0
        if dets is not None:
            angles = angles_from(dets)
            m3 = decode_3d(dets, fx_m, fy_m, fx_ref)
            used = [False] * len(gt_m)
            for i in np.argsort(-dets['scores']):
                rb = (float(dets['cx'][i]), float(dets['cy'][i]),
                      float(dets['w'][i]), float(dets['h'][i]), float(angles[i]))
                best, bj = 0.0, -1
                for j in range(len(gt_m)):
                    if used[j]:
                        continue
                    v = rotated_iou(rb, gt_m[j])
                    if v > best:
                        best, bj = v, j
                hit = labelled and bj >= 0 and best >= args.iou_thresh
                if hit:
                    used[bj] = True
                if labelled:
                    n_tp += int(hit)
                    n_fp += int(not hit)
                    if hit:
                        ious.append(best)
                        sizes_tp.append((float(m3['W'][i]), max(gt_m[bj][2], gt_m[bj][3])))
                rows.append(dict(
                    rbox=rb, score=float(dets['scores'][i]), matched=bool(hit),
                    iou=float(best) if labelled else None,
                    depth_m=float(m3['depth'][i]), W_m=float(m3['W'][i]),
                    L_m=float(m3['L'][i]), H_m=float(m3['H'][i]),
                    p_unit=float(m3['p_unit'][i])))
            n_fn += used.count(False)
        else:
            n_fn += len(gt_m)

        # ---- per-image figure ------------------------------------------
        fig = plt.figure(figsize=(13.5, 6.2), facecolor=BG)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(disp, cmap="gray", vmin=0, vmax=255)
        ax1.axis("off")
        for g in gt_m:
            draw_rbox(ax1, g, C_GT)
        for r in rows:
            col = (C_HIT if r["matched"] else C_MISS) if labelled else C_PRED
            draw_rbox(ax1, r["rbox"], col, label=f"{r['score']:.2f}")
        legend = ("label (green) · matched (blue) · false positive (red)"
                  if labelled else "no label for this frame — predictions only")
        ax1.set_title(f"{jpg.name} — {legend}", fontsize=10, color=INK)

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        pts = []
        for r in rows:
            rb = r["rbox"]
            c = build_corners_3d(rb[0], rb[1], r["depth_m"], r["W_m"], r["L_m"],
                                 r["H_m"], rb[4], fx_m, fy_m, cx_m, cy_m)
            pts.append(c)
            col = (C_HIT if r["matched"] else C_MISS) if labelled else C_PRED
            ax2.add_collection3d(Poly3DCollection(
                [[c[j] for j in f] for f in FACES],
                alpha=0.16, edgecolor=col, facecolor=col))
            kind = "unit" if r["p_unit"] >= 0.5 else "parcel"
            ax2.text(*c.mean(0),
                     f"{r['W_m']*100:.0f}x{r['L_m']*100:.0f}x{r['H_m']*100:.0f} cm\n"
                     f"{r['depth_m']:.2f} m · {kind} {r['p_unit']:.2f}",
                     color=INK, fontsize=7.5, ha="center", va="center")
        if pts:
            P = np.concatenate(pts, 0)
            lo, hi = P.min(0), P.max(0)
            pad = 0.08 * np.maximum(hi - lo, 1e-3)
            lo, hi = lo - pad, hi + pad
            ax2.set_xlim(lo[0], hi[0]); ax2.set_ylim(lo[1], hi[1]); ax2.set_zlim(lo[2], hi[2])
            ax2.set_box_aspect(tuple(hi - lo))
        else:
            ax2.text2D(0.5, 0.5, "no detection above threshold",
                       transform=ax2.transAxes, ha="center", color="#8a8984")
        ax2.set_xlabel("X (m)", fontsize=8)
        ax2.set_ylabel("Y (m)", fontsize=8)
        ax2.set_zlabel("Z depth (m)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.set_title(f"3D — W×L×H need no calibration; depth assumes "
                      f"fx={args.fx_native:.0f} px", fontsize=10, color=INK)
        fig.tight_layout()
        fig.savefig(out_dir / f"{jpg.stem}.png", dpi=args.dpi, facecolor=BG)
        plt.close(fig)

        records[jpg.name] = dict(labelled=labelled, n_labels=len(gt_m),
                                 detections=rows)
        if labelled:
            sheet.append((jpg.stem, disp, gt_m, rows))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(frames)}")

    # ---- outputs -------------------------------------------------------
    (out_dir / "predictions.json").write_text(json.dumps(dict(
        checkpoint=str(ckpt), checkpoint_epoch=epoch, data=str(data),
        conf=args.conf, nms_iou=args.nms_iou, iou_thresh=args.iou_thresh,
        fx_native_assumed=args.fx_native,
        note=("rbox = (cx, cy, w, h, angle_rad) in 640x640 square-padded model "
              "pixels; W/L/H in metres, fx-free; depth_m assumes fx_native"),
        frames=records), indent=1))

    pages = contact_sheets(sheet, out_dir) if sheet else 0

    prec = n_tp / max(n_tp + n_fp, 1)
    rec = n_tp / max(n_tp + n_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    lines = [
        f"checkpoint: {ckpt} (epoch {epoch})",
        f"frames: {len(frames)} ({n_lab} labelled, {len(frames) - n_lab} unlabelled)",
        f"detection on the {n_lab} labelled frames (conf>={args.conf}, IoU>={args.iou_thresh}):",
        f"  TP={n_tp}  FP={n_fp}  FN={n_fn}",
        f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}",
    ]
    if ious:
        lines.append(f"  mean IoU (TP) = {np.mean(ious):.3f}")
    if len(sizes_tp) > 2:
        st = np.array(sizes_tp)
        lines += [
            f"predicted long edge W on matched units: median {np.median(st[:, 0]):.3f} m, "
            f"p5 {np.percentile(st[:, 0], 5):.3f}, p95 {np.percentile(st[:, 0], 95):.3f}",
            f"  corr(W, human 2D long edge in px) = {np.corrcoef(st[:, 0], st[:, 1])[0, 1]:+.3f}"
            f"   (no metric ground truth exists for these frames; a size read from the "
            f"image would correlate clearly positively)",
        ]
    (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {len(frames)} overlays, {pages} contact sheet(s), predictions.json, "
          f"metrics.txt -> {out_dir}")


if __name__ == "__main__":
    main()
