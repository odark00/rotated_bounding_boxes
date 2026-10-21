# custom_3d_model

RTMDet-Rotated-style detector that predicts oriented 2D boxes plus metric
depth/size for parcels ("packages") in warehouse-ceiling-camera imagery, from
a single grayscale image and known camera intrinsics.

## What the model predicts, and why it is parameterised this way

The head emits, densely per anchor point: `cls, cx, cy, w, h, sin_a, cos_a,
depth_canon, w3d_px, l3d_px, h3d_px`. Two design choices there deserve
explanation, because they are the difference between a model that can learn
this task and one that cannot.

### 1. Boxes use the long-edge convention

`(w, h, angle)` always has **`w >= h`**, with `angle` the image-plane direction
of the *long* edge, wrapped to `[0, pi)`.

The obvious alternative — tie `w`/`angle` to the parcel's own "width" axis — is
unlearnable. The same pixels are described equally well by `(w, h, angle)` and
`(h, w, angle + 90deg)`, and which of the two a label carries depends on the
parcel's internal axis naming, which no image reveals. **41% of TASLER parcels
project to a near-square top face** (long/short < 1.15), so the old target
flipped by 90 degrees essentially at random across half the dataset. That is
the direct cause of the ~41 deg angle MAE the earlier runs reported — the
detector was fine, the label was a coin flip.

A genuinely square top face still has no defined orientation, but that
ambiguity is now a property of the shape rather than of the labelling, and it
is handled where it belongs: the angle loss is weighted by elongation
`(w-h)/(w+h)`, and `evaluate.py` reports angle error separately over boxes
elongated enough for an angle to exist.

### 2. Depth and height are predicted in intrinsics-free units

The TASLER cameras span 11 lens models with **fx from 411 to 1100 px** (a 2.68x
spread) at a fixed 640x640 render size. A parcel at 4 m under a 1100 px lens
projects to exactly the same pixels as one at 1.5 m under a 411 px lens, so
metric depth and metric height are *not functions of the image*. Measured on
the labels, `log(H3d_metres)` is predicted from apparent size with **R² =
0.000** — no signal whatsoever. The old head regressed metric metres anyway, so
the best it could do was emit the dataset average, which is exactly what its
0.062 m H MAE (~37% of the median height) reflects.

Both become well posed in units the image actually determines — an "apparent
size in pixels", which is what a camera physically measures:

```
depth_canon = depth * FX_REF / fx      # depth under a reference lens
w3d_px      = W_3d  * fx / depth       # apparent long  dimension, pixels
l3d_px      = L_3d  * fy / depth       # apparent short dimension, pixels
h3d_px      = H_3d  * fy / depth       # apparent height,          pixels
```

(`R²` from apparent size rises 0.494 → 0.630 for depth and 0.000 → 0.313 for
height.) The metric answer is then recovered by exact geometry from the known
intrinsics, with no learned scale factor anywhere:

```
depth = depth_canon * fx / FX_REF
W_3d  = w3d_px * depth / fx      # long horizontal dimension
L_3d  = l3d_px * depth / fy      # short horizontal dimension
H_3d  = h3d_px * depth / fy
```

**The metric dimensions come from their own heads, not from the 2D box edges.**
Deriving them instead as `W_3d = w_px * depth / fx` assumes the top face is
fronto-parallel with its whole edge at the centre depth; measured against the
labels that approximation costs a mean **8.8 mm and up to 51 cm** on W — a
systematic floor no amount of training removes. Feeding GT through the split
formulation and back recovers the labels to ~1e-7 m, i.e. float precision.
Separating them also keeps `w`/`h` a pure 2D detection quantity — what NMS, IoU
matching and the angle share — while the reported dimensions come from heads
supervised directly on the metric labels.

`model.decode_metric()` is the single implementation of that conversion;
`predict.py` and `evaluate.py` both call it, so inference and evaluation cannot
drift apart. **This makes `--intrinsics` load-bearing at inference** — without
it `predict.py` assumes the checkpoint's `FX_REF` camera and warns, and every
metric number scales linearly with how wrong that assumption is.

`FX_REF` (800 px, roughly the median TASLER fx) is stored in the checkpoint
hparams. Changing it changes the meaning of the depth head.

## Architecture (`model.py`, `losses.py`)

- **Backbone**: grayscale ResNet (`resnet18`/`resnet34`/`resnet50`), C3/C4/C5.
- **Neck**: lightweight FPN → P3/P4/P5 (strides 8/16/32), `--d_model` wide
  (default 128; 256 was ~1.15x slower for no measured gain on a single-class
  problem).
- **Head**: shared depthwise-conv tower per level. Depth is parameterised
  log-uniformly over `--depth_canon_range`, so head resolution is uniform in
  *relative* error and matches the log-space depth loss.
- **Assignment**: top-k (k=9) centre-distance per GT per level, size-gated so
  each GT trains only the level matching its scale. Level ranges are
  `(0,48) (48,96) (96,inf)` px, set from the actual label distribution (median
  long edge 41 px): the previous `(0,64) (64,128) (128,inf)` sent **83%** of
  objects to P3 and starved P4/P5 at 9%/10%, leaving two thirds of the head
  capacity training on almost nothing. The new split is 65/24/11.
- **Losses**: focal (cls) + **GWD** (Gaussian Wasserstein Distance on the
  oriented box) + L1 on centre/log-size + elongation-weighted angle L1 +
  log-space L1 on `depth_canon` and the three apparent extents. Every term is
  normalised by the
  number of positive anchors, so a 40-parcel image and a 2-parcel image
  contribute in proportion to their content rather than equally.

GWD scores the box as a whole, so centre, size and angle trade off the way IoU
does, and a near-square box has a near-isotropic covariance — its ambiguous
angle barely moves the loss. It is built directly from `(sin 2a, cos 2a)`,
never halving the angle, so there is no non-differentiable wrap point.

## Datasets (`dataset.py`)

`ParcelDataset` reads a sharded `shard_XXXXXXXX/{idx}.gray.png` +
`{idx}.json` layout and supports **two label schemas**, auto-detected per
sample:

- **`TASLER/`** — original schema, exactly one labelled parcel per image,
  described via world-frame scene geometry (`camera_pose`, `scene`,
  `object_dimensions_m`) reconstructed into camera-frame 3D box corners at
  load time. 197 images.
- **`TASLER_MULTI/`** — `"objects": [...]`, **multiple** labelled objects per
  image, each already carrying precomputed camera-frame corners. 3500 images,
  ~17 labelled packages per image.
- **`TASLER_PACK/`** — same `"objects"` schema, but built to look like the real
  `packstuecke/` footage: a ceiling camera over a forklift carrying one
  palletised shipping unit. ~2.6k images, 1-4 labelled units per image. See
  [TASLER_PACK](#tasler_pack-packstuecke-style-data-build_packstuecke_datasetpy)
  below.

Both produce the same per-sample target dict (`centers`, `depths`, `sizes`,
`rboxes`, `boxes_3d`, `rot6d`, `R`, ...) with a leading dimension of `N`
objects. `collate_fn` keeps targets as a list since `N` varies per image.

### Augmentation

- `augment=True` — photometric only (contrast/brightness/noise), geometry and
  intrinsics untouched.
- `geom_augment=True` — a random **quarter-turn rotation**, with every label
  rotated to match (`rot90_target`). Only the four *proper* rotations of the
  square are used, never a mirror: a reflection would map the object rotation
  matrix `R` to an improper matrix (det = -1) and silently corrupt `rot6d`.
  This is an exact 4x enlargement of an otherwise ~3.7k-image dataset.

`train.py` builds two dataset objects over the same file list and splits by
index, so augmentation applies to the training subset only. (The previous
single-dataset + `random_split` arrangement could not express that, which is
why augmentation was never actually switched on.)

### TASLER_MULTI: how it was built (`build_multi_dataset.py`)

`build_multi_dataset.py` converts `../synthetic_dataset_blender`'s
`gray_images/` + `labels/` output (Blender camera convention, `parcels[]` with
`dimensions` + `bbox_3d_camera`) into `TASLER_MULTI/`: axis-flips corners to
OpenCV convention, reorders Blender's `bound_box` indices into the
bottom(0-3)/top(4-7) layout, filters on in-frame centre / min projected size /
plausible metric dims / depth, caps each image at the 30 largest objects, and
symlinks the images. Re-run any time the render output changes:

```bash
python3 build_multi_dataset.py
```

### TASLER_PACK: packstuecke-style data (`build_packstuecke_dataset.py`)

`packstuecke/` holds 2877 real frames from the deployed Dachser overhead
cameras with 1932 hand-drawn rotated boxes (`*.txt`: `class cx cy w h
angle_deg`, normalised). They look nothing like `TASLER_MULTI`, and the gap is
what this dataset closes:

| | `TASLER_MULTI` | `packstuecke/` (real) |
|---|---|---|
| scene | loose parcels on a floor | forklift carrying a palletised unit |
| labelled object | each carton | the whole shipping unit |
| objects per image | ~17 | 1 (96%), up to 4 |
| long edge, as a share of the frame | a few % | p5 0.30 / median 0.47 / p95 0.72 |
| framing | full frame | a rectified crop, 63% of them pasted top-left into a 1280x1280 black canvas |
| tone | wide-open studio lighting | mean 60/255, sigma 34 over the valid pixels |

Generation is two steps:

1. **Render** — `DS_SCENE=packstuecke` in
   `../synthetic_dataset_blender/script.py` adds a second scene mode. It
   un-hides the rigged forklift that was already in `scene.blend` (its
   collection is not linked into the scene, so its objects sit outside the view
   layer and no hide flag alone can make it render), poses it so its blades
   carry a pallet, builds the unit on that pallet as one of four layouts
   (`single` / `column` / `stack` / `bulk`), and wraps most multi-piece units in
   stretch film — which is what makes a stack of loose cartons read as ONE
   shipping unit, the thing the real annotations enclose. `fx` is solved
   backwards per frame from how big the unit has to come out
   (`px = fx * long_edge / depth`), then clamped so the implied crop of the
   source sensor stays between 700 px and the full sensor width: a longer
   effective focal length *is* the real pipeline's crop.

   The label is the whole unit's OBB, **pallet included** — that is the
   shipping unit a dimensioning system has to measure. It is exported through
   an invisible proxy cuboid, so the existing label writer and converter need
   no special case.

   ```bash
   cd ../synthetic_dataset_blender
   DS_SCENE=packstuecke DS_OUT_TAG=_pack DS_NUM_IMAGES=2700 \
     DS_PACK_IMAGES_PER_MODEL=300 ./run.sh bg     # ~3 h on the RTX 4070 Laptop
   ```

2. **Frame + convert** — `build_packstuecke_dataset.py` applies the 2D framing
   and photometry of the real images and writes `TASLER_PACK/`. Crop, isotropic
   scale and translate are exact pinhole operations, so the 3D labels never
   move and only `K` does; the tone curve is fitted to both the mean *and* the
   standard deviation of the real valid-pixel distribution, because matching the
   mean alone leaves the render flat (sigma ~18 against the real ~34). Objects
   the crop pushed into the black padding are dropped — there is no image
   there, so keeping them would be a false negative the detector is punished
   for. The 3D corner conventions and sanity filters are imported from
   `build_multi_dataset.py` rather than restated, so the two synthetic datasets
   cannot drift apart.

   ```bash
   python3 build_packstuecke_dataset.py --check_dir /tmp/pk_check --check_n 24
   ```

**How close it came.** Measured over 500 units / 300 images of `TASLER_PACK`
against the real labels and frames:

| | real `packstuecke/` | `TASLER_PACK` |
|---|---|---|
| units per image | 1.02 | 1.07 |
| valid-pixel mean / sigma | 59.5 / 32.1 | 65.6 / 27.7 |
| valid fraction of the canvas | 0.733 | 0.731 |
| long edge, share of the frame (p5/med/p95) | 0.30 / 0.47 / 0.72 | 0.39 / 0.60 / 0.94 |
| top-face aspect, median | 1.43 | 1.24 |
| ... share near-square (< 1.15) | 0.062 | 0.183 |

Count, tone and padding match. The two that do not are worth knowing before
using this for anything but training:

- **Units come out ~28% too big.** `PACK_FRAC` is a target on the *render*
  frame, and the converter's crop then enlarges the object again relative to
  the valid rect. Multiply `PACK_FRAC` by ~0.78 to close it.
- **Footprints are too square** (18% near-square against the real 6%), because
  `PACK_PALLET_SIZES` is dominated by 1.2x1.0-ish footprints and the OBB
  includes the pallet. Re-weight `PACK_PALLET_WEIGHTS` towards 1.2x0.8, drop
  the square 1.2x1.2, and add a long (1.6x0.8) pallet. This one matters: a
  square top face makes the angle target ambiguous, which is exactly the defect
  the long-edge convention was introduced to fix.

Both need a re-render (~3 h); the constants are left as they are so the
generator still reproduces the `TASLER_PACK/` currently on disk.

**Known tension, on purpose.** `TASLER_MULTI` labels every carton of a stack
individually; `TASLER_PACK` labels the stack as one unit and leaves its cartons
unlabelled. Training on both therefore sends conflicting signals about what
"an object" is. The cues that separate the two cases are the scale (a unit
fills 30-70% of the frame, a `TASLER_MULTI` parcel a few percent), the pallet
and forklift under it, and the stretch film around it. Score the two with
`evaluate.py --only_root` before assuming it came out cleanly.

## Training (`train.py`)

```bash
python3 train.py --epochs 50 --run_name my_run --ckpt_dir checkpoints_multi/my_run
```

To train on the packstuecke-style data as well, add it to `--data_dir` — the
loader pools any number of roots:

```bash
python3 train.py --epochs 50 --run_name rtmdet_pack \
  --data_dir TASLER_MULTI TASLER TASLER_PACK \
  --ckpt_dir checkpoints_multi/rtmdet_pack
```

That is the full default configuration: `--data_dir TASLER_MULTI TASLER`,
`--batch_size 8`, `--backbone resnet50`, `--d_model 128`, `--lr 1e-3`,
augmentation on, **50 epochs**, ~1.25 h on an RTX 4070 Laptop.

Key args: `--backbone {resnet18,resnet34,resnet50}`, `--d_model`,
`--num_head_convs`, `--num_queries` (cap on labelled objects per image),
`--depth_canon_range` (must cover `depth * FX_REF / fx`, which spans
0.71–11.31 here — note this is **canonical**, not metric, depth),
`--no_augment`, `--patience` (EarlyStopping on `val/loss`). Prints
`BEST_CKPT: <path>` at the end.

### Stopping and continuing

`--resume <ckpt>` continues an **interrupted** run exactly where it stopped:
Lightning restores weights, optimizer, LR-schedule and epoch counter, and model
hyperparameters are read back from the checkpoint (so architecture flags on the
command line are ignored — pass the same `--epochs` as the original run or the
cosine schedule will not line up).

```bash
python3 train.py --epochs 50 --run_name my_run --ckpt_dir checkpoints_multi/my_run \
  --resume checkpoints_multi/my_run/last.ckpt
```

`ModelCheckpoint` writes `last.ckpt` on every epoch alongside the three best by
`val/loss`, so an interrupted run always has an exact resume point. (Resuming
from a top-k checkpoint instead costs whatever epochs happened after it.)

`--init_ckpt <path>` is the *other* thing: it warm-starts model weights but
builds a fresh optimizer and a new warmup/cosine schedule sized to
`--epochs`/`--lr` — for pushing an already-**finished** run further, not for
resuming an interrupted one.

Logging goes to TensorBoard, falling back to `csv_logs/` if `tensorboard`
isn't installed rather than aborting a multi-hour run at setup.

### Where the speed came from

50 epochs went from ~1.93 h to ~1.25 h (**1.54x**) while also training on 4x
augmented data:

| change | effect |
|---|---|
| vectorised anchor assignment | 22x faster per (image, level); the old Python double loop forced a GPU sync per GT per level, ~400 per step at ~17 objects/image |
| `channels_last` | ~10%, free — matches the NHWC tensor-core kernels AMP dispatches to |
| `d_model` 256 → 128 | ~1.15x |

The vectorised assigner is bit-exact against the original (0 mismatches over
256k anchors). `torch.compile` was measured and **did not help** on this GPU
("not enough SMs" for max-autotune), so it is not used.

## Evaluation (`evaluate.py`)

Reproduces the same held-out val split `train.py` used (same
`--data_dir`/`--img_size`/`--n_val` (default 600 images)/seed=42 permutation), decodes with
`predict.py`'s `decode_level`/`nms_rotated` and `model.decode_metric`, and
matches predictions to GT by **greedy rotated-IoU matching** (score-sorted
predictions each claim the best available unmatched GT above `--iou_thresh`).

```bash
python3 evaluate.py --ckpt checkpoints_multi/my_run/my_run-42.ckpt \
  --conf 0.4 --iou_thresh 0.5 --max_parcels 30
```

Reports precision/recall/F1, mean IoU, and MAE for depth (absolute and
relative), W/L/H and angle. Two conventions to read carefully:

- **W is the long horizontal dimension, L the short one.** GT dims are sorted
  the same way before comparison, matching the long-edge box convention.
- **angle MAE is reported twice**: over all TPs, and over the `--elong_ratio`
  (default 1.15) subset whose top face is elongated enough for an orientation
  to exist. The all-TP number is partly a measure of how many square parcels
  the split contains; the elongated number is the one that reflects the angle
  head.

`--out_csv` dumps per-sample rows.

`--only_root TASLER_PACK` restricts the held-out split to the images that came
from one source root. Pass the **same** `--data_dir` the model was trained with
and add the filter — the seed-42 permutation runs over the pooled file list, so
passing a single root in `--data_dir` would build a different, and
training-contaminated, val set:

```bash
python3 evaluate.py --ckpt <ckpt> --data_dir TASLER_MULTI TASLER TASLER_PACK \
  --only_root TASLER_PACK
```

## Model comparison (`run_experiments.py`)

Trains several backbone/width variants for 50 epochs each, evaluates each, and
picks the winner by detection F1 (tie-broken by summed W/L/H MAE):

```bash
python3 run_experiments.py            # all variants
python3 run_experiments.py --only resnet50_baseline
```

Variants: `resnet18_fast`, `resnet34_light`, `resnet50_baseline`,
`resnet50_wide` (d_model 256). Writes `experiment_logs/{train,eval}_<name>.log`
and a running `experiment_summary.json`.

## Inference / visualization (`predict.py`)

```bash
python3 predict.py --ckpt <path> --image img.png --intrinsics FX FY CX CY
python3 predict.py --ckpt <path> --folder imgs/ --out_dir predictions/ --intrinsics FX FY CX CY
```

Draws 2D rotated boxes plus a 3D box plot. `--intrinsics` takes fx fy cx cy
**in original-image pixels** and is required for trustworthy metric output (see
above) — without it the script prints a warning and assumes the checkpoint's
`FX_REF` camera. `--conf` sets the score threshold and `--nms_iou` the
rotated-IoU duplicate threshold.

`--max_labels` (default 8) controls the 3D panel: at or below it every box is
annotated with its W/L/H, above it the boxes carry indices matching the table
printed to stdout. Per-box dimension text is unreadable once a scene holds 30
packages, which is routine here. The 3D axes are scaled from the actual corner
extents with `set_box_aspect`, so a cube renders as a cube — `Poly3DCollection`
does not drive matplotlib's autoscaling, so without that the boxes came out as
slabs on arbitrary axis ranges.

NMS uses true rotated-box IoU (`geometry.rotated_iou`, a Sutherland-Hodgman
polygon clip with a circumradius short-circuit, no shapely dependency). The
previous centre-distance approximation merged distinct parcels sitting close
together — common in images holding up to 30 packed packages — while leaving
genuine duplicates of long thin boxes alive.

### Calling it from Python

`predict()` now takes an already-loaded module, not a checkpoint path, so a
batch of images pays the load cost once:

```python
from predict import load_model, predict, detect
model, device = load_model("checkpoints_multi/my_run/last.ckpt")
predict(model, "img.png", conf_thresh=0.5, intrinsics=(fx, fy, cx, cy),
        out_path="out.png", device=device)
```

`detect(model, x, conf, nms_iou)` returns the raw NMS'd detection dict without
plotting anything, for use in your own pipeline; pair it with
`model.decode_metric` and `predict.angles_from` to get metres and radians.

## Compatibility note

Checkpoints trained before this revision are **not loadable**: the head's
output keys changed (`depth`/`height3d` → `depth_canon` + `w3d_px`/`l3d_px`/
`h3d_px`), and the box
and angle conventions changed with them. Metrics from `experiment_summary.json`
entries predating this revision are likewise not comparable — in particular
their W/L MAE used the old per-axis convention and their angle MAE was
dominated by the labelling ambiguity described above. Retrain rather than
fine-tune.

## Legacy / dead code

`matcher.py` and `visualizer_gt.py` belong to a pre-migration DETR3D-style
model (Hungarian-matched queries) and a flat `images/`+`labels/` dataset
layout with Blender-convention JSON — not wired into any current script.
`checkpoints/parcel3d*.ckpt` and everything under `checkpoints_multi/` predate
this revision (see compatibility note).
