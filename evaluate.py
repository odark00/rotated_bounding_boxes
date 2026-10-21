"""
Evaluation script for the RTMDet-Rotated parcel detector.

Runs a trained checkpoint over a held-out split of the TASLER dataset and
reports detection precision/recall/F1 (matched by rotated-box IoU) plus
regression error (MAE) on true positives for depth, W/L/H and angle.

Decoding (score threshold, per-level gather, rotated NMS) is imported straight
from predict.py, and metric conversion from model.decode_metric, so evaluation
always matches single-image inference exactly.

The val split is reproduced with the same (--data_dir, --img_size,
--n_val, seed=42) permutation train.py uses, so the default invocation
evaluates on the same held-out images a matching training run never saw.

Conventions (see dataset._rbox_from_top_face):
  * boxes use the long-edge convention, so the predicted W is the LONGER
    horizontal dimension and L the shorter; GT dimensions are sorted the same
    way before comparison.
  * angle MAE is reported twice: over all true positives, and over the subset
    whose top face is elongated enough for an orientation to exist at all
    (--elong_ratio). A square parcel has no defined angle and averaging it into
    the headline number just measures how many squares the dataset contains.

Usage:
    python evaluate.py --ckpt checkpoints/run/run-42.ckpt --data_dir TASLER_MULTI TASLER
    python evaluate.py --ckpt ... --conf 0.3 --iou_thresh 0.5 --out_csv eval.csv
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader, Subset

from dataset import ParcelDataset, collate_fn
from losses import RTMDetLitModule
from model import decode_3d
from geometry import rotated_iou
from predict import decode_level, nms_rotated, angles_from, STRIDES, DET_KEYS


def slice_level_outs(level_outs, b):
    return [{k: v[b:b + 1] for k, v in out.items()} for out in level_outs]


def decode_image(level_outs_b, score_thresh, nms_iou):
    all_dets = {k: [] for k in DET_KEYS}
    for out, stride in zip(level_outs_b, STRIDES):
        d = decode_level(out, stride, score_thresh)
        if d is not None:
            for k in all_dets:
                all_dets[k].append(d[k])
    if not all_dets['scores']:
        return None
    return nms_rotated({k: np.concatenate(v) for k, v in all_dets.items()}, nms_iou)


def angle_diff_deg(a, b):
    """Circular difference between two angles in [0, pi) (rect has 180deg symmetry)."""
    d = abs(a - b) % math.pi
    return math.degrees(min(d, math.pi - d))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(args):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = RTMDetLitModule.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)

    img_size = args.img_size or model.hparams.img_size
    fx_ref = float(model.hparams.fx_ref)

    full = ParcelDataset(args.data_dir, img_size=img_size, max_parcels=args.max_parcels)
    n_val = max(1, min(args.n_val, len(full) - 1))
    perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(42)).tolist()
    val_idx = perm[:n_val]

    if args.only_root:
        # Score one source dataset WITHOUT changing the split: the seed-42
        # permutation runs over the pooled file list, so passing a single root
        # in --data_dir would produce a different (and training-contaminated)
        # val set. Build the pooled split as usual, then keep the part of it
        # that came from this root.
        keep = str(Path(args.only_root).resolve())
        val_idx = [i for i in val_idx
                   if str(full.samples[i][0].resolve()).startswith(keep)]
        if not val_idx:
            raise SystemExit(f"no validation images under {args.only_root!r}; "
                             f"is it one of --data_dir {args.data_dir}?")
        print(f"[eval] --only_root {args.only_root}: {len(val_idx)} of {n_val} "
              f"validation images")
    val_ds = Subset(full, val_idx)

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    n_tp = n_fp = n_fn = 0
    tp_iou, depth_err, W_err, L_err, H_err = [], [], [], [], []
    depth_rel, ang_err, ang_err_elong = [], [], []
    W_rel, H_rel = [], []
    type_hits = {0: [], 1: []}
    rows = []

    for imgs, targets in loader:
        imgs = imgs.to(device)
        level_outs = model.model(imgs)

        for b in range(imgs.shape[0]):
            tgt = targets[b]
            fx, fy = tgt['K'][0, 0].item(), tgt['K'][1, 1].item()
            n_gt = tgt['num_parcels']
            gt_rboxes = tgt['rboxes'].tolist()
            gt_depths = tgt['depths'][:, 0].tolist()
            gt_sizes = tgt['sizes'].tolist()          # (n_gt, 3) = (H, W, L)
            gt_types = tgt['obj_type'].tolist()
            gt_used = [False] * n_gt

            dets = decode_image(slice_level_outs(level_outs, b), args.conf, args.nms_iou)

            if dets is not None:
                angles = angles_from(dets)
                m3 = decode_3d(dets, fx, fy, fx_ref)
                depth_m, W3d, L3d, H3d = m3['depth'], m3['W'], m3['L'], m3['H']
                p_unit = m3['p_unit']

                # Greedy match: predictions sorted by score, each claims the best
                # unmatched GT (by rotated IoU) if it clears the threshold.
                for i in np.argsort(-dets['scores']):
                    pred_rbox = (float(dets['cx'][i]), float(dets['cy'][i]),
                                 float(dets['w'][i]), float(dets['h'][i]), float(angles[i]))

                    best_iou, best_j = 0.0, -1
                    for j in range(n_gt):
                        if gt_used[j]:
                            continue
                        iou = rotated_iou(pred_rbox, tuple(gt_rboxes[j]))
                        if iou > best_iou:
                            best_iou, best_j = iou, j

                    row = dict(stem=tgt['stem'], score=float(dets['scores'][i]))

                    if best_j >= 0 and best_iou >= args.iou_thresh:
                        gt_used[best_j] = True
                        n_tp += 1
                        tp_iou.append(best_iou)

                        gt_depth = gt_depths[best_j]
                        gt_H, gt_w, gt_l = gt_sizes[best_j]
                        # long-edge convention: compare long-to-long, short-to-short
                        gt_W, gt_L = max(gt_w, gt_l), min(gt_w, gt_l)

                        pd, pW, pL, pH = (float(depth_m[i]), float(W3d[i]),
                                          float(L3d[i]), float(H3d[i]))

                        depth_err.append(abs(pd - gt_depth))
                        depth_rel.append(abs(pd - gt_depth) / max(gt_depth, 1e-6))
                        W_err.append(abs(pW - gt_W))
                        L_err.append(abs(pL - gt_L))
                        H_err.append(abs(pH - gt_H))
                        W_rel.append(abs(pW - gt_W) / max(gt_W, 1e-6))
                        H_rel.append(abs(pH - gt_H) / max(gt_H, 1e-6))
                        gt_t = int(gt_types[best_j])
                        type_hits[gt_t].append(int(float(p_unit[i]) >= 0.5) == gt_t)

                        ad = angle_diff_deg(float(angles[i]), gt_rboxes[best_j][4])
                        ang_err.append(ad)
                        gw, gh = gt_rboxes[best_j][2], gt_rboxes[best_j][3]
                        if gw / max(gh, 1e-6) >= args.elong_ratio:
                            ang_err_elong.append(ad)

                        row.update(matched=True, iou=best_iou, gt_depth=gt_depth,
                                   gt_W=gt_W, gt_L=gt_L, gt_H=gt_H, angle_err_deg=ad,
                                   pred_depth=pd, pred_W=pW, pred_L=pL, pred_H=pH)
                    else:
                        n_fp += 1
                        row.update(matched=False, iou=best_iou if best_j >= 0 else None,
                                   gt_depth=None, gt_W=None, gt_L=None, gt_H=None,
                                   angle_err_deg=None, pred_depth=None, pred_W=None,
                                   pred_L=None, pred_H=None)
                    rows.append(row)

            for j in range(n_gt):
                if gt_used[j]:
                    continue
                n_fn += 1
                gt_H, gt_w, gt_l = gt_sizes[j]
                rows.append(dict(stem=tgt['stem'], score=None, matched=False, iou=None,
                                 gt_depth=gt_depths[j], gt_W=max(gt_w, gt_l),
                                 gt_L=min(gt_w, gt_l), gt_H=gt_H, angle_err_deg=None,
                                 pred_depth=None, pred_W=None, pred_L=None, pred_H=None))

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def mae(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    print(f"\nEvaluated {len(val_ds)} validation images "
          f"(conf>={args.conf}, IoU>={args.iou_thresh}, nms_iou={args.nms_iou})")
    print(f"  TP={n_tp}  FP={n_fp}  FN={n_fn}")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
    print(f"  mean IoU (TP)   = {mae(tp_iou):.3f}")
    print(f"  depth MAE (TP)  = {mae(depth_err):.4f} m   ({100*mae(depth_rel):.1f}% rel)")
    print(f"  W MAE (TP)      = {mae(W_err):.4f} m   (long horizontal dim)")
    print(f"  L MAE (TP)      = {mae(L_err):.4f} m   (short horizontal dim)")
    print(f"  H MAE (TP)      = {mae(H_err):.4f} m")
    if W_rel:
        print(f"  W median rel err = {np.median(W_rel) * 100:.1f}%   "
              f"H median rel err = {np.median(H_rel) * 100:.1f}%")
    for t, name in ((0, "parcel"), (1, "palletised unit")):
        if type_hits[t]:
            print(f"  type accuracy, {name:15s} = {np.mean(type_hits[t]):.3f}  "
                  f"({len(type_hits[t])} TPs)")
    print(f"  angle MAE (TP)  = {mae(ang_err):.2f} deg  (all {len(ang_err)})")
    print(f"  angle MAE elong = {mae(ang_err_elong):.2f} deg  "
          f"(the {len(ang_err_elong)} TPs with long/short >= {args.elong_ratio})")

    if args.out_csv and rows:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-sample results -> {args.out_csv}")

    return dict(precision=precision, recall=recall, f1=f1, mean_iou=mae(tp_iou),
                depth_mae=mae(depth_err), depth_rel=mae(depth_rel), W_mae=mae(W_err),
                L_mae=mae(L_err), H_mae=mae(H_err), angle_mae=mae(ang_err),
                angle_mae_elong=mae(ang_err_elong))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", type=str, nargs="+", default=["TASLER_MULTI", "TASLER"])
    p.add_argument("--n_val", type=int, default=600,
                   help="number of held-out validation images; must match train.py --n_val")
    p.add_argument("--img_size", type=int, default=None,
                   help="defaults to the checkpoint's img_size hyperparameter")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.4,
                   help="score threshold, same meaning as predict.py --conf")
    p.add_argument("--nms_iou", type=float, default=0.3,
                   help="rotated-IoU above which two detections are duplicates")
    p.add_argument("--iou_thresh", type=float, default=0.5,
                   help="rotated IoU required to count a detection as a true positive")
    p.add_argument("--elong_ratio", type=float, default=1.15,
                   help="long/short ratio above which a box's angle is well defined")
    p.add_argument("--out_csv", type=str, default=None,
                   help="optional path to dump per-sample results")
    p.add_argument("--only_root", type=str, default=None,
                   help="restrict the held-out split to images from this "
                        "--data_dir root (e.g. TASLER_PACK), for per-domain "
                        "numbers on the same split the model was trained with")
    p.add_argument("--max_parcels", type=int, default=30,
                   help="cap on labelled objects per image; should match the training run's --num_queries")
    args = p.parse_args()
    evaluate(args)
