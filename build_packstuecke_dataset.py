"""
Convert the `DS_SCENE=packstuecke` Blender renders into a TASLER-style sharded
dataset (`TASLER_PACK/`) that dataset.py's ParcelDataset reads unchanged.

Source (per image), written by synthetic_dataset_blender/script.py:
  gray_images_pack/img_XXXX.png     640x640 grayscale, ideal pinhole, no border
  labels_pack/img_XXXX.json         camera_intrinsics + parcels[] with
                                    bbox_3d_camera (8 corners, Blender camera
                                    frame: x right, y up, z back)

Output (per image):
  TASLER_PACK/shard_XXXXXXXX/{idx:08d}.gray.png   640x640, packstuecke framing
  TASLER_PACK/shard_XXXXXXXX/{idx:08d}.json       TASLER_MULTI schema

What this step adds on top of the render - the 2D framing of the real
`packstuecke/` images, kept here rather than in Blender so it can be re-tuned
without re-rendering:

  1. Framing. Every real image is a rectified CROP of the source sensor, and
     63% of them are then pasted top-left into a 1280x1280 black canvas at
     1200x1000 (or 1000x1200), leaving hard black bands on the right and
     bottom. Measured on 600 of them: the valid region is an axis-aligned
     rectangle at the origin, not a rotated one - the content inside looks
     rotated only because the camera roll varies. Both variants are
     reproduced at 640x640, which is the model's input size, so no further
     resize happens in the data loader.

     Crop + isotropic scale + translate are exact pinhole operations, so the
     3D labels are untouched and only K moves:
         fx' = fx * s,  cx' = (cx - x0) * s + pad_x   (same for y)

  2. Photometry. The real frames are dark and flat - mean 60/255 over the
     valid pixels (p5 37, p95 88), sigma 34 - with strong vignetting, sensor
     grain and JPEG artefacts. The render's exposure is matched to a sample
     from that distribution by a clamped gain applied in linear light, which
     is what changing a camera's exposure actually does; vignette, grain and a
     JPEG round-trip follow.

The 3D corner conventions (Blender camera frame -> OpenCV, bottom0-3/top4-7
cyclic order) and the per-object sanity filters are imported from
build_multi_dataset.py rather than restated, so the two synthetic datasets can
never drift apart.

Usage:
    python3 build_packstuecke_dataset.py                  # default paths
    python3 build_packstuecke_dataset.py --limit 50 --check_dir /tmp/pk_check
"""

import argparse
import io
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from build_multi_dataset import (
    BOTTOM_IDX, TOP_IDX, DIM_RANGES, MIN_DEPTH_M, MAX_DEPTH_M,
    MIN_RBOX_AREA_PX, blender_cam_to_opencv, rbox_from_top_face,
)

HERE = Path(__file__).resolve().parent
SRC_ROOT = Path("/home/daryna/code_ws/study/3D_detection/synthetic_dataset_blender")

OUT_RES = 640          # canvas side; also the model's input size
SHARD_SIZE = 250
MAX_OBJECTS_PER_IMAGE = 12

# --- framing (measured on packstuecke/*.jpg) -------------------------------
PAD_PROB = 0.63        # share pasted into a square canvas with black bands
LANDSCAPE_PROB = 0.69  # of those, the share that are 1200x1000 not 1000x1200
VALID_LONG = 1200 / 1280.0     # valid rect, as a fraction of the canvas
VALID_SHORT = 1000 / 1280.0
CROP_FRAC = (0.86, 1.0)        # how much of the render the crop keeps
CROP_JITTER = 0.30             # crop offset, as +-this share of the slack
CORNER_CLIP_PROB = 0.15        # a rectification trapezoid cuts one corner
CORNER_CLIP = (0.06, 0.28)     # ... this far along each of its two edges

# --- photometry (same source) ---------------------------------------------
TARGET_MEAN = (37.0, 60.0, 88.0)   # p5 / median / p95 of the valid-pixel mean
TARGET_STD = (16.3, 33.8, 59.9)    # ... and of its standard deviation
GAIN_CLAMP = (0.35, 3.0)
CONTRAST_CLAMP = (0.7, 4.0)
VIGNETTE_K = (0.0, 0.38)
GRAIN_SIGMA = (0.8, 4.0)
JPEG_QUALITY = (70, 92)
GAMMA = 2.2


def _lognorm(p5, med, p95, rng):
    s = (math.log(p95 / med) + math.log(med / p5)) / (2 * 1.645)
    return med * math.exp(rng.gauss(0.0, s))


def plan_framing(rng):
    """Pick one framing. Returns (crop_w, crop_h, dst_w, dst_h, pad) in pixels
    of the render (crop_*) and of the output canvas (dst_*, pad)."""
    R = OUT_RES
    if rng.random() < PAD_PROB:
        if rng.random() < LANDSCAPE_PROB:
            dst_w, dst_h = round(R * VALID_LONG), round(R * VALID_SHORT)
        else:
            dst_w, dst_h = round(R * VALID_SHORT), round(R * VALID_LONG)
    else:
        dst_w = dst_h = R

    # crop a window of the render with the destination's aspect ratio
    u = rng.uniform(*CROP_FRAC)
    ar = dst_w / dst_h
    if ar >= 1.0:
        crop_w = R * u
        crop_h = crop_w / ar
    else:
        crop_h = R * u
        crop_w = crop_h * ar
    if crop_h > R:
        crop_h = R * u
        crop_w = crop_h * ar
    return crop_w, crop_h, dst_w, dst_h


def apply_framing(img, K, rng):
    """Crop / scale / paste one render. Returns (PIL image, new K dict)."""
    R = OUT_RES
    crop_w, crop_h, dst_w, dst_h = plan_framing(rng)
    # The scene builder already puts the unit near the optical axis, so the
    # crop is centred with only a modest jitter - sampling the offset uniformly
    # over a 190 px letterbox slack threw the labelled unit out of frame.
    x0 = (R - crop_w) * (0.5 + rng.uniform(-CROP_JITTER, CROP_JITTER))
    y0 = (R - crop_h) * (0.5 + rng.uniform(-CROP_JITTER, CROP_JITTER))
    s = dst_w / crop_w              # == dst_h / crop_h by construction

    win = img.resize((dst_w, dst_h), Image.LANCZOS,
                     box=(x0, y0, x0 + crop_w, y0 + crop_h))
    canvas = Image.new("L", (R, R), 0)
    canvas.paste(win, (0, 0))

    K2 = dict(fx=K["fx"] * s, fy=K["fy"] * s,
              cx=(K["cx"] - x0) * s, cy=(K["cy"] - y0) * s,
              width=R, height=R)
    return canvas, K2, (dst_w, dst_h)


def clip_corner(arr, valid_wh, rng):
    """Black out a triangle at one corner of the valid rect - the trapezoidal
    edge undistortion leaves on a minority of the real frames."""
    w, h = valid_wh
    fx_ = rng.uniform(*CORNER_CLIP) * w
    fy_ = rng.uniform(*CORNER_CLIP) * h
    corner = rng.randrange(4)
    ys, xs = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    if corner == 0:      # top-left
        m = (xs / fx_ + ys / fy_) < 1.0
    elif corner == 1:    # top-right
        m = ((w - 1 - xs) / fx_ + ys / fy_) < 1.0
    elif corner == 2:    # bottom-left
        m = (xs / fx_ + (h - 1 - ys) / fy_) < 1.0
    else:                # bottom-right
        m = ((w - 1 - xs) / fx_ + (h - 1 - ys) / fy_) < 1.0
    m &= (xs < w) & (ys < h)
    arr[m] = 0
    return arr


def photometric(arr, valid_wh, K, rng):
    """Exposure + contrast match, vignette, grain, JPEG round-trip.

    Matching the mean alone is not enough: the render comes out flat (sigma
    ~18) where the real footage is high-contrast (sigma ~34) - a lit pallet of
    pale cardboard against a dark concrete floor, with the shipping labels
    clipping to white. The tone curve here is therefore fitted to both the mean
    AND the standard deviation of the real valid-pixel distribution.
    """
    w, h = valid_wh
    x = arr.astype(np.float32) / 255.0
    lin = np.power(np.clip(x, 0.0, 1.0), GAMMA)

    # exposure: a clamped gain in linear light
    cur = float(np.power(lin[:h, :w].mean(), 1.0 / GAMMA) * 255.0)
    t_mean = _lognorm(*TARGET_MEAN, rng)
    gain = 1.0 if cur < 1e-3 else (t_mean / cur) ** GAMMA
    lin *= min(max(gain, GAIN_CLAMP[0] ** GAMMA), GAIN_CLAMP[1] ** GAMMA)

    # vignette, falling off from the principal point (linear light, as on a
    # real lens: it is lost illumination, not a tone curve)
    k = rng.uniform(*VIGNETTE_K)
    if k > 0.01:
        ys, xs = np.mgrid[0:arr.shape[0], 0:arr.shape[1]].astype(np.float32)
        r2 = (((xs - K["cx"]) / max(w, 1)) ** 2
              + ((ys - K["cy"]) / max(h, 1)) ** 2)
        lin *= np.clip(1.0 - k * r2, 0.05, 1.0)

    out = np.power(np.clip(lin, 0.0, 1.0), 1.0 / GAMMA) * 255.0

    # contrast: a linear stretch about the valid region's own mean, so the
    # final (mean, sigma) land on the sampled target. Highlights clip, which is
    # what the white labels do on the real sensor too.
    t_std = _lognorm(*TARGET_STD, rng)
    # Two passes: the first stretch clips at 0/255, which pulls sigma back
    # below the target, so it is re-fitted on the clipped result.
    for _ in range(2):
        mu = float(out[:h, :w].mean())
        sd = float(out[:h, :w].std())
        if sd < 1e-3:
            break
        c = min(max(t_std / sd, CONTRAST_CLAMP[0]), CONTRAST_CLAMP[1])
        out = np.clip(mu + c * (out - mu) + (t_mean - mu), 0.0, 255.0)

    out = out + np.random.normal(0.0, rng.uniform(*GRAIN_SIGMA), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # the black bands are added back after the grain, as in the real pipeline:
    # they are padding, not sensor output
    out[:, w:] = 0
    out[h:, :] = 0
    if rng.random() < CORNER_CLIP_PROB:
        out = clip_corner(out, valid_wh, rng)

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG",
                              quality=rng.randint(*JPEG_QUALITY))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


MIN_INSIDE_CORNERS = 1      # of the 4 top-face corners, inside the valid rect;
                            # a unit that fills the frame can easily have
                            # three of them outside it and still be obvious


def convert_objects(parcels, K, valid_wh):
    """Blender-camera parcels -> TASLER_MULTI `objects[]`, using the framed K.

    An object is kept only if it is actually inside the VALID rect, not merely
    somewhere on the canvas: the black padding carries no image, so a unit that
    the crop pushed into it is invisible and would be a false negative the
    detector gets punished for.
    """
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    width, height = valid_wh

    objects = []
    for p in parcels:
        dims = p.get("dimensions")
        corners_b = p.get("bbox_3d_camera")
        if not dims or not corners_b or len(corners_b) != 8:
            continue
        width_m, length_m, height_m = dims[0], dims[1], dims[2]
        if not all(DIM_RANGES[k][0] <= v <= DIM_RANGES[k][1]
                   for k, v in (("width_m", width_m), ("length_m", length_m),
                                ("height_m", height_m))):
            continue

        corners_cv = blender_cam_to_opencv(corners_b)
        boxes_3d_cam = np.concatenate([corners_cv[BOTTOM_IDX],
                                       corners_cv[TOP_IDX]], axis=0)
        depth = float(boxes_3d_cam.mean(axis=0)[2])
        if not (MIN_DEPTH_M <= depth <= MAX_DEPTH_M):
            continue

        (px, py), w_px, h_px, _ = rbox_from_top_face(
            boxes_3d_cam[4:8], fx, fy, cx, cy)
        if not (0.0 <= px <= width and 0.0 <= py <= height):
            continue
        if w_px * h_px < MIN_RBOX_AREA_PX:
            continue
        top = boxes_3d_cam[4:8]
        tz = np.clip(top[:, 2], 1e-6, None)
        tu = top[:, 0] / tz * fx + cx
        tv = top[:, 1] / tz * fy + cy
        inside = int(np.sum((tu >= 0) & (tu <= width)
                            & (tv >= 0) & (tv <= height)))
        if inside < MIN_INSIDE_CORNERS:
            continue

        objects.append(dict(
            name=p.get("name", ""),
            type=p.get("type", "palletised_load"),
            object_dimensions_m=dict(width_m=width_m, length_m=length_m,
                                     height_m=height_m),
            boxes_3d_cam=boxes_3d_cam.round(6).tolist(),
            _area=w_px * h_px,
        ))

    objects.sort(key=lambda o: -o["_area"])
    objects = objects[:MAX_OBJECTS_PER_IMAGE]
    for o in objects:
        del o["_area"]
    return objects


def convert_one(label_path, img_path, rng):
    label = json.loads(label_path.read_text())
    ci = label["camera_intrinsics"]
    if int(ci["width"]) != OUT_RES or int(ci["height"]) != OUT_RES:
        raise ValueError(f"{label_path}: expected a {OUT_RES}px square render, "
                         f"got {ci['width']}x{ci['height']}")

    img = Image.open(img_path).convert("L")
    framed, K, valid_wh = apply_framing(img, ci, rng)
    arr = photometric(np.asarray(framed), valid_wh, K, rng)

    objects = convert_objects(label.get("parcels", []), K, valid_wh)
    if not objects:
        return None, None

    meta = dict(
        camera_intrinsics=dict(
            fx=K["fx"], fy=K["fy"], cx=K["cx"], cy=K["cy"],
            width=K["width"], height=K["height"],
            source_camera_model=ci.get("source_camera_model"),
            source_camera_type=ci.get("source_camera_type"),
        ),
        num_objects=len(objects),
        objects=objects,
        packstuecke=dict(
            valid_region=[0, 0, valid_wh[0], valid_wh[1]],
            **(label.get("packstuecke") or {}),
        ),
    )
    return Image.fromarray(arr), meta


def draw_check(img, meta, path):
    """Overlay the projected 3D boxes, for eyeballing a sample."""
    from PIL import ImageDraw
    ci = meta["camera_intrinsics"]
    rgb = img.convert("RGB")
    d = ImageDraw.Draw(rgb)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for o in meta["objects"]:
        c = np.array(o["boxes_3d_cam"])
        z = np.clip(c[:, 2], 1e-6, None)
        u = c[:, 0] / z * ci["fx"] + ci["cx"]
        v = c[:, 1] / z * ci["fy"] + ci["cy"]
        for a, b in edges:
            d.line([(u[a], v[a]), (u[b], v[b])], fill=(255, 70, 70), width=2)
    rgb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(SRC_ROOT / "gray_images_pack"))
    ap.add_argument("--labels", default=str(SRC_ROOT / "labels_pack"))
    ap.add_argument("--out", default=str(HERE / "TASLER_PACK"))
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check_dir", default="")
    ap.add_argument("--check_n", type=int, default=0)
    args = ap.parse_args()

    src_img = Path(args.images)
    src_lab = Path(args.labels)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(src_lab.glob("img_*.json"))
    if args.limit:
        label_paths = label_paths[:args.limit]
    print(f"Found {len(label_paths)} source labels in {src_lab}")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    check_dir = Path(args.check_dir) if args.check_dir else None
    if check_dir:
        check_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_image = 0
    skipped_no_objects = 0
    obj_counts = []
    shard_dir = None

    for lp in label_paths:
        ip = src_img / f"{lp.stem}.png"
        if not ip.exists():
            skipped_no_image += 1
            continue
        img, meta = convert_one(lp, ip, rng)
        if meta is None:
            skipped_no_objects += 1
            continue

        if written % SHARD_SIZE == 0:
            shard_dir = out_root / f"shard_{written:08d}"
            shard_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{written:08d}"
        img.save(shard_dir / f"{stem}.gray.png")
        (shard_dir / f"{stem}.json").write_text(json.dumps(meta))
        if check_dir and written < args.check_n:
            draw_check(img, meta, check_dir / f"{stem}.png")

        obj_counts.append(meta["num_objects"])
        written += 1

    print(f"Wrote {written} samples to {out_root}")
    print(f"Skipped: {skipped_no_image} (no image), "
          f"{skipped_no_objects} (no valid objects after filtering)")
    if obj_counts:
        a = np.array(obj_counts)
        print(f"Objects/image: min={a.min()} max={a.max()} mean={a.mean():.2f} "
              f"total={a.sum()}")


if __name__ == "__main__":
    main()
