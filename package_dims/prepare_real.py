"""Real frames of cam_000CDF0AD90F -> test set framed exactly like the synthetic data.

    raw 2560x2048 --undistort--> pinhole --resize x0.5--> 1280x1024 --top-left on black--> 1280x1280

Writes <out>/shard_00000000/{idx:08d}.gray.png + .json. The json carries the same
`camera_intrinsics` block as the synthetic labels, an empty `objects` list (the 3D pose of
the real units is unknown) and a `ground_truth` block parsed from the dataset Readme.md:
    ground_truth: {package, L_m, W_m, H_m, dmc_corners_px (in the 1280x1280 image), source}
L/W/H are the overall unit size, pallet included, H from the floor.

    python prepare_real.py
"""
import argparse
import glob
import json
import os
import re

import cv2
import numpy as np

from camera import Camera, pad_square

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_readme(path):
    """Readme.md -> {image_stem: {package, L_m, W_m, H_m}}."""
    out, cur = {}, None
    num = lambda line: float(re.search(r"([\d.]+)\s*m", line).group(1))  # noqa: E731  first value wins
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("## Package"):
            cur = {"package": int(line.split()[-1])}
        elif cur is None:
            continue
        elif line.startswith("* Length"):
            cur["L_m"] = num(line)
        elif line.startswith("* Width"):
            cur["W_m"] = num(line)
        elif line.startswith("* Height"):
            cur["H_m"] = num(line)
        elif line.startswith("* raw_"):
            out[line[2:].strip()] = dict(cur)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-dir", default=os.path.join(os.path.dirname(HERE), "cam_000CDF0AD90F"))
    ap.add_argument("--out", default=os.path.join(HERE, "datasets", "REAL_CAM_000CDF0AD90F"))
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--canvas", type=int, default=1280)
    args = ap.parse_args()

    full, cam = Camera(args.cam_dir, 1.0), Camera(args.cam_dir, args.scale)
    gt = parse_readme(os.path.join(args.cam_dir, "Readme.md"))
    shard = os.path.join(args.out, "shard_00000000")
    os.makedirs(shard, exist_ok=True)
    for i, png in enumerate(sorted(glob.glob(os.path.join(args.cam_dir, "raw_*.png")))):
        stem = os.path.basename(png)[:-4]
        img = cv2.resize(full.undistort_image(cv2.imread(png, cv2.IMREAD_GRAYSCALE)), cam.size,
                         interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(shard, f"{i:08d}.gray.png"), pad_square(img, args.canvas))
        raw = json.load(open(png[:-4] + ".json"))
        g = dict(gt.get(stem, {}), source=stem)
        pk = raw.get("codescanner_result", {}).get("packages", [])
        if pk and "corners" in pk[0]:
            c = np.array([[p["x"], p["y"]] for p in pk[0]["corners"]])
            g["dmc_corners_px"] = (full.undistort_points(c) * args.scale).round(2).tolist()
        label = {"camera_intrinsics": cam.intrinsics_json(args.canvas), "num_objects": 0, "objects": [],
                 "camera_pose": {"height_m": cam.mount_h, "tilt_deg": 0.0, "roll_deg": 0.0},
                 "valid_region": [0, 0, cam.size[0], cam.size[1]], "ground_truth": g}
        with open(os.path.join(shard, f"{i:08d}.json"), "w") as f:
            json.dump(label, f, indent=1)
        print(f"{i:08d}  {stem}  {g.get('L_m')} x {g.get('W_m')} x {g.get('H_m')}  dmc={'dmc_corners_px' in g}")


if __name__ == "__main__":
    main()
