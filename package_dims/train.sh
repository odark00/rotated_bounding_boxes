#!/usr/bin/env bash
# Train custom_3d_model (single-image 3D parcel detector) on the synthetic data of this camera.
# NOT started automatically — run it yourself when the GPU is free:
#     ./train.sh                    # fine-tune from rtmdet_pack_v3 (recommended)
#     INIT= ./train.sh              # from scratch
# Extra args are passed through to train.py, e.g.  ./train.sh --epochs 30 --batch_size 4
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/daryna/code_ws/study/3D_detection/custom_3d_model}"
# one or more dataset folders (space separated, pooled): path-traced v4 + the 505 images of v3
DATA="${DATA:-$HERE/datasets/SYNTH_CAM_000CDF0AD90F_v4 $HERE/datasets/SYNTH_CAM_000CDF0AD90F_v3}"
RUN="${RUN:-cam5_v4}"
INIT="${INIT-$MODEL_DIR/checkpoints_multi/rtmdet_pack_v3_best/rtmdet_pack_v3-32.ckpt}"  # warm start; INIT= for scratch

cd "$MODEL_DIR"
python3 train.py \
  --data_dir $DATA \
  --run_name "$RUN" \
  --ckpt_dir "$HERE/checkpoints/$RUN" \
  --img_size 640 \
  --n_val 300 \
  --epochs 50 \
  ${INIT:+--init_ckpt "$INIT"} \
  "$@"
