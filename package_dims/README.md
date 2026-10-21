# package_dims — 3D size of packages from the overhead camera cam_000CDF0AD90F

Single overhead monochrome camera (JAI GO-5000M, 2560x2048, mounted 5.70 m, looking straight
down), no depth sensor. Goal: length x width x height of the shipping unit (pallet included),
CPU inference, offline.

Plan: train the existing single-image 3D detector
`~/code_ws/study/3D_detection/custom_3d_model` (oriented box + metric W/L/H from one grayscale
image + intrinsics) on **synthetic images of this exact camera**, and test on the real frames.

## Layout

```
camera.py                 camera model: intrinsics, distortion, project/backproject, square padding
synth/generate.py         procedural renderer -> synthetic dataset (CPU, ~0.1 s/image)
synth/generate_v2.py      v2: z-buffered procedural renderer, part-based scenes (CPU)
synth/generate_v3.py      path-traced generator (Blender Cycles, GPU, via synth/blender_render.py) -> dataset v4
synth/units_v3.py         shipping units beyond cartons (drums, sacks, tyres, film-wrapped ...), soft cartons, film
synth/generate_v5.py      v5: diverse cameras / floors / surroundings / layouts / units (Blender, via blender_render_v5.py)
synth/env_v5.py           v5 cameras (LM8/12/16 calibrations), procedural floors, warehouse structures, people, vehicles
prepare_real.py           real frames -> test set in the same framing + Readme ground truth
tools/visualize_labels.py draw the 3D labels on images (sanity check)
tools/eval_real.py        trained checkpoint on the real frames -> box + L x W x H in cm vs the Readme truth
train.sh                  launches custom_3d_model/train.py on the synthetic data (not run yet)
assets/background_median.png   empty floor = per-pixel median of the 19 real frames
datasets/
  SYNTH_CAM_000CDF0AD90F/      3000 synthetic images + 3D ground truth   (training)
  SYNTH_CAM_000CDF0AD90F_v3/   505 path-traced images, v2 scenes (run stopped early)   (training)
  SYNTH_CAM_000CDF0AD90F_v4/   3000 path-traced images, all unit kinds (v4)           (training)
  SYNTH_V5_DIVERSE/            3000 path-traced images, varied camera / floor / scene (v5), nothing from cam_000CDF0AD90F
  REAL_CAM_000CDF0AD90F/       19 real frames + Readme sizes             (test only)
```

## Data format (= custom_3d_model `objects` schema, same as TASLER_MULTI / TASLER_PACK)

```
shard_XXXXXXXX/{idx:08d}.gray.png   1280x1280: undistorted 1280x1024 frame top-left, black padding below
shard_XXXXXXXX/{idx:08d}.json
  camera_intrinsics  fx, fy, cx, cy, width, height    (pinhole, for the 1280x1280 image)
  objects[]          one per shipping unit
    type                 "palletised_load" | "parcel"
    object_dimensions_m  width_m = |c1-c0|, length_m = |c3-c0|, height_m (from the floor)
    boxes_3d_cam         8 corners, OpenCV camera frame (x right, y down, z forward), metres,
                         bottom 0-3 on the floor, top 4-7; the box encloses pallet + load
    synth                extra, ignored by the loader: kind, top_face_3d, pallet_top_3d,
                         dmc_3d (DataMatrix corners), load/pallet sizes
  camera_pose        height_m 5.702, tilt 0, roll 0
  valid_region       [0, 0, 1280, 1024]
```

Real test frames have `objects: []` and a `ground_truth` block instead:
`{package, L_m, W_m, H_m, dmc_corners_px, source}` (`dmc_corners_px` already in the 1280x1280
undistorted image). Frames of package 8 have no DMC.

**Images are undistorted.** custom_3d_model ignores lens distortion, so the synthetic data is
pinhole and every real frame must go through the same `prepare_real.py` framing before
inference (undistort -> x0.5 -> pad to 1280x1280). Intrinsics of that image:

```
fx 1277.03  fy 1277.39  cx 632.45  cy 504.42
```

## What the synthetic data contains

Rendered with the real intrinsics + mounting height over the real empty floor:

| | share | notes |
|---|---|---|
| pallet loads | ~65 % | EUR 1.2x0.8 (60 %), 1.2x1.0, random; load ≈ pallet / overhanging panel / small carton on pallet; carton grid, cardboard or white panel textures, 50 % stretch film |
| loose cartons | ~20 % | 0.3-1.4 m, no pallet |
| cages (gitterbox) | ~15 % | open top, mesh walls |
| units per image | 0 (8 %), 1, 2 (~10 %) | 0 = negatives (floor / pallet truck only) |
| height | 0.15-2.25 m | real Readme range 0.73-2.17 m |
| extras | | DMC label (16x16, 94.8 mm) on 90 % of tops, pallet trucks, floor shadows, sun patch, gain/gamma/noise/blur |

Exact labels: the unit box is the OBB of pallet + load (matches the Readme convention, e.g.
package 6: L 1.6 = load overhang, W 0.8 = pallet).

## Synthetic v2 (`synth/generate_v2.py`)

Same schema/framing/label convention as v1, more realistic scenes:

- z-buffered renderer (per-pixel plane depth): forks hidden under pallet decks, gaps between
  cartons, see-through cage mesh all occlude correctly
- one light direction per scene: Lambert shading of every face, side faces darker towards the
  floor, directional + contact floor shadows
- geometry built from parts: block pallets (deck boards, cross boards, 9 blocks), loads made of
  individual cartons (uniform / ragged-top stacks, mixed columns with cartons askew on top, single
  big box, plywood crate, panel, strapped machine, small item), gitterbox cages with posts and rim
- pallet truck engaged under ~70 % of pallets (forks inside, heel, tilted drawbar + loop handle),
  as in nearly all real frames; truck is not part of the label box
- carton SKUs (brand text / oval logos sized in metres, small print, flap seam + tape, worn boxes),
  stretch film as an alpha overlay with edge-biased crinkle highlights, straps
- sun patch from the bottom-right edge in 80 % of images (real: 15/19 frames)

Tone on valid pixels (mean / sigma): real 65.9 / 36.8, v1 66.8 / 43.5, v2 65.9 / 37.1.
~1.5 CPU-s per image.

```bash
nice -n 19 python3 synth/generate_v2.py --n 3000 --workers 8 --seed 0   # -> datasets/SYNTH_CAM_000CDF0AD90F_v2
```

## Synthetic v3 (`synth/generate_v3.py` + `synth/blender_render.py`) — path-traced

Same scenes, labels, schema and framing as v2 (scene builders imported from `generate_v2.py`), but the
image is rendered with **Blender Cycles on the GPU** instead of the OpenCV quad rasteriser:

- physically based light: mostly ambient + a grid of ceiling lamps (sometimes a few lamps, a soft dock-door
  light from the bottom right), giving soft real shadows, contact occlusion under / inside pallets and
  between cartons, light bounced from the floor
- the floor is a shadow catcher textured with the real empty floor, so shadows land on the real background
  and objects are composited over it with antialiased edges
- materials: cardboard with bulging faces, darker handled rims and scuffs, rounded edges (bevel) on
  everything; painted truck; galvanised cage mesh; stretch film = clear plastic rounded over the load edges
  with patchy crinkles that catch lamp glints + v2's edge-biased milky creases
- pallet truck rebuilt from round parts (tapered forks, pump, wheels, drawbar, rounded-triangle handle)
- camera: linear, gain/gamma/sun patch/vignetting as v2, sensor noise at the real level (sigma ~0.8)

Tone on valid pixels (mean / sigma): real 65.9 / 36.2, v3 64.0 / 35.0. Blender projection checked
against `Camera.project` (< 0.1 px). ~2 s per image (RTX 4070 laptop, 2 Blender processes).
Needs Blender >= 4.2 (tested 5.2.2); the first run compiles the OptiX kernels (~3 min, cached).

```bash
# v3 was made with the plain v2 scenes; generate_v3.py now builds the v4 scenes (see below) and writes to _v4
# resumes after an interruption (skips finished images); --samples / --device CUDA|CPU / --blender PATH
```

## Synthetic v4 (`synth/generate_v3.py` + `synth/units_v3.py`) — all kinds of shipping units

Same renderer, schema, framing and label convention as v3 (box = OBB of everything incl. pallet / carrier / film),
richer scenes. A cross-dock does not only see cartons:

| share of units | what |
|---|---|
| ~60 % | v2 cartons / pallet loads / cages, upgraded: stretch film draped over the real load outline (clear 70 %, bubble wrap, black / white opaque, tinted), soft cartons in 35 % (rounded edges and corners, bulging sides, sagging / domed tops, dents), edge protectors + top sheet + straps |
| ~32 % | sacks (paper / PE, interlocked layers), big bags with loops, beverage pallets, crate stacks on dollies, roll containers, machines / furniture under film, octabins, pallet collars, polybags (loose or heaped), slatted crates, IBCs, buckets / canisters, long goods on bearers |
| ~8 % (rare) | round goods: steel / plastic drums, cable drums (standing / lying), coils, tyres and wheels on rims, paper reels, carpet rolls, mailing tubes |

- smooth curved surfaces (per-corner normals), identical items share one texture
- shipping label = sticker with the DataMatrix on the highest flat spot, on top of the film (exact `dmc_3d`);
  ~73 % of units carry one (cages, crate dollies, roll containers, tubes, slatted lids: none)
- `synth` json extras: `kind` (e.g. `stack`, `sacks`, `tyres`), `film` (style or null), `soft`
- tone on valid pixels (mean / sigma): real 65.9 / 36.8, v4 previews 67 / 37; ~2 s per image (RTX 4070 laptop)

```bash
nice -n 10 python3 synth/generate_v3.py --n 3000 --workers 6 --seed 0   # -> datasets/SYNTH_CAM_000CDF0AD90F_v4
```

## Synthetic v5 (`synth/generate_v5.py` + `synth/env_v5.py` + `synth/blender_render_v5.py`) — diverse scenes

v1-v4 are all "this camera over this floor": one real median floor image, one camera, 0-2 isolated units, so the
images look alike. v5 changes everything per image and takes **nothing from cam_000CDF0AD90F**, so its 19 frames stay
an independent test (no real floor image, not its camera model 51185017, no tone / noise / sun-patch matching to its
frames, DMC label size random 75-115 mm instead of the 94.8 mm fitted on them, no "truck under 70 % of pallets").

| | v5 |
|---|---|
| camera | JAI GO-5000M + KOWA LM8HC (20 %, 2.6-4.0 m) / LM12HC (40 %, 3.9-6.3 m, model 51185020) / LM16HC (40 %, 5.3-8.0 m); median factory calibration of the shipped model + unit jitter; framed like `prepare_real.py` incl. the black undistortion corners; intrinsics per image in the json |
| floor | procedural and rendered (lit, glossy, shadowed): concrete, worn, polished, epoxy, dark painted, asphalt, tiles; slab joints, cracks, repairs, stains, wet patches, tyre marks, wear paths; lane lines, pallet bays with numbers, hatched zones, arrows, zone codes (worn); dock levellers; dark bay / trailer / yard beyond a dock wall |
| surroundings (never labelled) | walls with dock doors (open / half / closed, seals, bollards), columns with guards, guard rails, pallet racks with loads on the beams, roller / belt conveyors, forklifts, electric pallet trucks, hand pallet trucks, people (hi-vis, helmets, walking, carrying), empty pallets and stacks, bins, cones, bollards, cables, sheets, film rolls, trolleys, desks, overhead pipes |
| layouts | isolated 17 % / rows 30 % (touching, tight or spaced, along the floor axis) / clusters 22 % / dock 12 % / loose parcels 9 % / empty 10 %; units cut by the image border (centre outside, unlabelled); up to 10 labelled units per image (mean ~3.7) |
| units | v4 units + double-stacked pallet loads, half / quarter pallets, loose parcels; 50 % size-scaled (footprint 0.85-1.2, height 0.8-1.25), 50 % re-toned (dark / bright goods); height <= min(2.6 m, camera height - 1.3 m) |
| light | hall (lamp grid above the camera), night (low ambient, lamp pools), day (skylights); sun beams through openings; daylight through open dock doors |
| image | auto exposure (frame median 30-95), gain, gamma, vignetting, blur, shot + read noise (stronger at night) |

`json["scene"]` keeps mode, lens, height, floor kind, light style and the number of cut units of every image.
~3.7 s per image (RTX 4070 laptop, 2 Blender processes, 6 CPU workers).

```bash
nice -n 10 python3 synth/generate_v5.py --n 3000 --workers 6 --seed 0      # -> datasets/SYNTH_V5_DIVERSE (resumes)
DATA="$PWD/datasets/SYNTH_V5_DIVERSE $PWD/datasets/SYNTH_CAM_000CDF0AD90F_v4" RUN=cam5_v5 ./train.sh  # v5 + v4
DATA="$PWD/datasets/SYNTH_V5_DIVERSE" RUN=v5only ./train.sh   # v5 only: fair test on the cam5 frames
```

## How to generate a dataset

All generators write the same schema (`shard_XXXXXXXX/{idx:08d}.gray.png` + `.json`, 250 images per shard) and are
run from this folder (`~/Documents/dachser/package_dims`). For new data use **v5**.

**Requirements**
- Python 3 with `numpy`, `opencv-python` (tested 2.4 / 5.0)
- v3-v5 (path-traced): Blender >= 4.2 on `PATH` (tested 5.2.2, `/snap/bin/blender`) or `--blender /path/to/blender`,
  and an NVIDIA GPU (OptiX). The first run compiles the OptiX kernels (~3 min, cached). Without a GPU: `--device CPU`
  (much slower).
- v5 also reads `../cam_models_calib/2025.10.23_Dachser_Shipped.txt` (lens calibrations); v1-v4 read
  `../cam_000CDF0AD90F/` and `assets/background_median.png`.
- Resources for v5 with the defaults: ~3 GB GPU memory, ~6-8 GB RAM, ~0.5 MB disk per image (3000 images ~1.5 GB),
  ~4 s per image on an RTX 4070 laptop (3000 images ~3.5 h).

**1. Preview a few images first** (seconds per image, into a throwaway folder) and look at them with the labels drawn:

```bash
python3 synth/generate_v5.py --n 24 --seed 5 --out /tmp/v5_preview
python3 tools/visualize_labels.py /tmp/v5_preview /tmp/v5_preview.jpg --n 24
```

`visualize_labels.py` draws every labelled unit: green = bottom face, red = top face, blue = verticals, yellow = size.
The box must sit on the unit; units cut by the image border have no box (by design).

**2. Generate the dataset** (runs for hours: start it in the background, low CPU priority, keep a log):

```bash
nohup nice -n 10 python3 synth/generate_v5.py --n 3000 --workers 6 --seed 0 \
    --out datasets/SYNTH_V5_DIVERSE > datasets/generate_synth_v5.log 2>&1 &
tail -f datasets/generate_synth_v5.log          # "25/3000  4.06 s/img  eta 201 min"
```

**3. Stop / resume / extend.** Stop it any time (Ctrl-C / `kill`). Run the *same* command again to continue: images
whose `.png` and `.json` already exist are skipped. Every image is determined by `(--seed, index)`, so a rerun
re-creates exactly the missing ones. More images later: same `--seed` and `--out` with `--start 3000 --n 1000`; an
independent set: another `--seed` into another `--out`. The temporary `<out>/_work/` folder (scene files, raw renders,
`blender_*.log`) is deleted at the end; `--keep-work` keeps it for debugging. If the log says
`Blender exited (codes ...)`, the reason is in `<out>/_work/blender_*.log`.

**4. Check the result** (counts, then a few random shards):

```bash
ls datasets/SYNTH_V5_DIVERSE/shard_*/*.png | wc -l      # -> 3000
python3 tools/visualize_labels.py datasets/SYNTH_V5_DIVERSE /tmp/v5_check.jpg --n 16 --start 1500
```

**5. Train on it** — see the v5 section above (`DATA="$PWD/datasets/SYNTH_V5_DIVERSE" RUN=v5only ./train.sh`).

`generate_v5.py` options:

| option | default | |
|---|---|---|
| `--n`, `--start` | 3000, 0 | number of images, index of the first one |
| `--seed` | 0 | dataset seed (image i uses scene seed `seed * 1000003 + i`) |
| `--out` | `datasets/SYNTH_V5_DIVERSE` | output folder |
| `--workers` | 6 | CPU processes building scenes (~4 CPU-s each) and finishing images |
| `--blender-procs` | 2 | Blender processes rendering at the same time (overlap CPU and GPU) |
| `--samples` | 48 | Cycles samples per pixel (adaptive + denoised); more = cleaner and slower |
| `--device` | OPTIX | OPTIX / CUDA / CPU |
| `--blender` | `blender` on PATH | Blender executable |
| `--canvas` | 1280 | side of the square output image (frame top-left, black padding below) |
| `--keep-work` | off | keep `<out>/_work` |

What a scene contains (shares of layouts, lenses, floors, lights ...) is set by the constants at the top of
`synth/generate_v5.py` (`MODES`, `MAX_LABELLED`) and `synth/env_v5.py` (`LENSES`, `FLOOR_KINDS`), and by the
probabilities in `make_scene()`.

Older versions (same output schema, all tied to cam_000CDF0AD90F and its real floor):

```bash
nice -n 19 python3 synth/generate.py    --n 3000 --workers 4 --seed 0   # v1: CPU rasteriser  -> SYNTH_CAM_000CDF0AD90F
nice -n 19 python3 synth/generate_v2.py --n 3000 --workers 8 --seed 0   # v2: CPU, z-buffered -> ..._v2
nice -n 10 python3 synth/generate_v3.py --n 3000 --workers 6 --seed 0   # v4: Blender, GPU    -> ..._v4
```

## Commands

```bash
# regenerate / more data (CPU, low priority; does not touch the GPU)
nice -n 19 python3 synth/generate.py --n 3000 --workers 4 --seed 0
nice -n 19 python3 synth/generate.py --n 3000 --seed 1 --out datasets/SYNTH_CAM_000CDF0AD90F_s1

# real test set
python3 prepare_real.py

# look at labels
python3 tools/visualize_labels.py datasets/SYNTH_CAM_000CDF0AD90F /tmp/synth.jpg --n 16
python3 tools/visualize_labels.py datasets/REAL_CAM_000CDF0AD90F  /tmp/real.jpg  --n 19

# train (when the GPU is free) — warm-starts from custom_3d_model rtmdet_pack_v3, data = v4 + v3 by default
./train.sh --epochs 10
DATA="datasets/SYNTH_CAM_000CDF0AD90F_v4" RUN=cam5_v4only ./train.sh   # other data / run name

# evaluate on the 19 real frames: package box + L x W x H in cm next to the Readme sizes (csv, json, overlays)
python3 tools/eval_real.py --ckpt checkpoints/cam5_v4/<best>.ckpt   # -> predictions/real_eval/
#   rtmdet_pack_v3 as is (not trained on this camera):   MAE L 55.1  W 30.2  H 84.3 cm
#   cam5_v4-08.ckpt (10 epochs, v4 3000 + v3 505 images): MAE L  8.6  W 12.8  H 23.8 cm, 19/19 detected
#   cam5_v4_e20-08.ckpt (+10 epochs from it, --lr 2e-4):   MAE L  8.3  W 10.0  H 21.5 cm  (lr 1e-3 -> NaN)
INIT= ./train.sh --epochs 80          # from scratch

# predict on the real test frames (CPU)
cd ~/code_ws/study/3D_detection/custom_3d_model
CUDA_VISIBLE_DEVICES= python3 predict.py --ckpt <best.ckpt> \
  --folder ~/Documents/dachser/package_dims/datasets/REAL_CAM_000CDF0AD90F/shard_00000000 \
  --out_dir ~/Documents/dachser/package_dims/predictions \
  --intrinsics 1277.03 1277.39 632.45 504.42
```

## Notes / known gaps

- Synthetic data is procedural, not photoreal: real tone is mean 66 / sigma 37 on valid pixels,
  synthetic 68 / 43 (3000-image set). Check with a few real frames after training;
  domain gap is the main risk.
- Height alternative that needs no learning: the DMC symbol is a fixed 94.8 mm, so
  `height = 5.702 - fx * 0.0948 / side_px` (undistorted px, full res) gave ±5 cm on the real
  frames. `synth` json keeps `dmc_3d` so a DMC-based head/check can be trained or validated too.
- The label size 94.8 mm was fitted on the same 19 frames — measure a real label to confirm.
