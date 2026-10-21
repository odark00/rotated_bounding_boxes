"""Draw the 3D unit boxes of a dataset (synthetic or real-test) onto its images.

    python tools/visualize_labels.py datasets/SYNTH_CAM_000CDF0AD90F out.jpg --n 16

green = bottom face (floor), red = top face, blue = verticals, yellow = W x L x H (m).
Real-test images have no 3D boxes; their Readme ground truth is printed instead.
"""
import argparse
import glob
import json

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    tiles = []
    for p in sorted(glob.glob(f"{args.root}/shard_*/*.gray.png"))[args.start:args.start + args.n]:
        d = json.load(open(p.replace(".gray.png", ".json")))
        ci = d["camera_intrinsics"]
        im = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        for o in d["objects"]:
            c = np.array(o["boxes_3d_cam"])
            uv = np.stack([ci["fx"] * c[:, 0] / c[:, 2] + ci["cx"], ci["fy"] * c[:, 1] / c[:, 2] + ci["cy"]], 1)
            uv = uv.round().astype(int)
            for a in range(4):
                cv2.line(im, tuple(uv[a]), tuple(uv[(a + 1) % 4]), (0, 200, 0), 3)
                cv2.line(im, tuple(uv[4 + a]), tuple(uv[4 + (a + 1) % 4]), (0, 0, 255), 3)
                cv2.line(im, tuple(uv[a]), tuple(uv[a + 4]), (255, 0, 0), 2)
            dm = o["object_dimensions_m"]
            cv2.putText(im, f"{dm['width_m']:.2f}x{dm['length_m']:.2f}x{dm['height_m']:.2f}", tuple(uv[4]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        g = d.get("ground_truth")
        if g:
            cv2.putText(im, f"GT {g.get('L_m')}x{g.get('W_m')}x{g.get('H_m')}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
        tiles.append(cv2.resize(im, (400, 400)))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
    cv2.imwrite(args.out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


if __name__ == "__main__":
    main()
