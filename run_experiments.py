"""
Train several RTMDet-Rotated variants on TASLER_MULTI (50 epochs each), then
evaluate each best checkpoint on the same held-out val split and report which
one wins on detection F1 and on size/depth regression error.

Usage:
    python run_experiments.py                 # run all variants
    python run_experiments.py --only resnet50_baseline
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIRS = ["TASLER_MULTI", "TASLER"]
EPOCHS = 50
COMMON = [
    "--data_dir", *DATA_DIRS,
    "--epochs", str(EPOCHS),
    "--batch_size", "8",
    "--num_queries", "30",
    "--num_workers", "8",
    # canonical-depth range, not metric: must cover depth * FX_REF / fx, which
    # spans 0.71-11.31 over this dataset (see model.py).
    "--depth_canon_range", "0.5", "14.0",
    "--patience", "15",
]

VARIANTS = [
    dict(name="resnet18_fast", backbone="resnet18", num_head_convs=2, d_model=128),
    dict(name="resnet34_light", backbone="resnet34", num_head_convs=2, d_model=128),
    dict(name="resnet50_baseline", backbone="resnet50", num_head_convs=2, d_model=128),
    dict(name="resnet50_wide", backbone="resnet50", num_head_convs=2, d_model=256),
]

LOG_DIR = ROOT / "experiment_logs"
LOG_DIR.mkdir(exist_ok=True)
SUMMARY_PATH = ROOT / "experiment_summary.json"


def run(cmd, log_path):
    print(f"\n=== RUNNING: {' '.join(cmd)} ===", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        for line in proc.stdout:
            f.write(line)
            f.flush()
            lines.append(line)
        proc.wait()
    return proc.returncode, lines


def parse_best_ckpt(lines):
    ckpt = None
    for line in lines:
        m = re.match(r"BEST_CKPT:\s*(.+)", line.strip())
        if m:
            ckpt = m.group(1).strip()
    return ckpt


def parse_eval_metrics(lines):
    text = "".join(lines)
    out = {}
    for key, pat in [
        ("precision", r"precision=([\d.]+)"),
        ("recall", r"recall=([\d.]+)"),
        ("f1", r"F1=([\d.]+)"),
        ("mean_iou", r"mean IoU \(TP\)\s*=\s*([\d.]+)"),
        ("depth_mae", r"depth MAE \(TP\)\s*=\s*([\d.]+)"),
        ("W_mae", r"W MAE \(TP\)\s*=\s*([\d.]+)"),
        ("L_mae", r"L MAE \(TP\)\s*=\s*([\d.]+)"),
        ("H_mae", r"H MAE \(TP\)\s*=\s*([\d.]+)"),
        ("angle_mae", r"angle MAE \(TP\)\s*=\s*([\d.]+)"),
        ("angle_mae_elong", r"angle MAE elong\s*=\s*([\d.]+)"),
        ("depth_rel", r"depth MAE \(TP\)\s*=\s*[\d.]+ m\s*\(([\d.]+)% rel\)"),
    ]:
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="run a single variant by name")
    args = ap.parse_args()

    variants = [v for v in VARIANTS if args.only is None or v["name"] == args.only]
    summary = {}
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())

    for v in variants:
        name = v["name"]
        t0 = time.time()
        ckpt_dir = ROOT / "checkpoints_multi" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable, "train.py",
            "--run_name", name,
            "--backbone", v["backbone"],
            "--num_head_convs", str(v["num_head_convs"]),
            "--d_model", str(v["d_model"]),
            "--ckpt_dir", str(ckpt_dir),
        ] + COMMON

        rc, lines = run(train_cmd, LOG_DIR / f"train_{name}.log")
        elapsed = time.time() - t0
        if rc != 0:
            summary[name] = dict(status="FAILED", train_seconds=elapsed)
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
            print(f"!!! {name} training FAILED (rc={rc}) — see experiment_logs/train_{name}.log")
            continue

        best_ckpt = parse_best_ckpt(lines)
        summary[name] = dict(status="TRAINED", train_seconds=elapsed, best_ckpt=best_ckpt,
                              backbone=v["backbone"], num_head_convs=v["num_head_convs"],
                              d_model=v["d_model"])
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

        if not best_ckpt:
            print(f"!!! {name}: could not find BEST_CKPT in training output, skipping eval")
            continue

        eval_cmd = [
            sys.executable, "evaluate.py",
            "--ckpt", best_ckpt,
            "--data_dir", *DATA_DIRS,
            "--conf", "0.4",
            "--iou_thresh", "0.5",
            "--max_parcels", "30",
        ]
        rc, elines = run(eval_cmd, LOG_DIR / f"eval_{name}.log")
        metrics = parse_eval_metrics(elines) if rc == 0 else {}
        summary[name]["status"] = "EVALUATED" if rc == 0 else "EVAL_FAILED"
        summary[name]["metrics"] = metrics
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        print(f"=== {name} done in {elapsed/60:.1f} min, metrics: {metrics} ===", flush=True)

    # ---- pick best ----
    scored = []
    for name, s in summary.items():
        m = s.get("metrics") or {}
        f1 = m.get("f1")
        if f1 is None:
            continue
        size_mae = sum(x for x in (m.get("W_mae"), m.get("L_mae"), m.get("H_mae")) if x is not None)
        # rank primarily by F1 (detection quality), tie-broken by lower size MAE
        scored.append((f1, -size_mae, name))
    scored.sort(reverse=True)

    if scored:
        best_name = scored[0][2]
        summary["_best"] = best_name
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        print(f"\n=== BEST MODEL: {best_name} ===")
        print(json.dumps(summary[best_name], indent=2))
    else:
        print("\nNo variant produced usable metrics.")


if __name__ == "__main__":
    main()
