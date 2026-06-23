# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
!pip install opencv-python-headless==4.10.0.84 --no-deps --force-reinstall

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install torch>=2.1 \
# MAGIC torchvision>=0.16 \
# MAGIC pytorch-lightning>=2.2 \
# MAGIC numpy \
# MAGIC pillow \
# MAGIC matplotlib \
# MAGIC scipy \
# MAGIC tensorboard

# COMMAND ----------

import torch
torch.__version__

# COMMAND ----------

!python3 -V

# COMMAND ----------

# MAGIC %sh
# MAGIC python visualizer_gt.py --name img_0000 --out_dir .

# COMMAND ----------

# MAGIC %sh
# MAGIC python train.py \
# MAGIC   --images /Volumes/playground/it_cl_terminal/ilo/data/blender_generated/images \
# MAGIC   --labels /Volumes/playground/it_cl_terminal/ilo/data/blender_generated/labels \
# MAGIC   --batch_size 8 --epochs 200 --num_queries 10

# COMMAND ----------

# MAGIC %sh
# MAGIC python predict.py \
# MAGIC   --ckpt checkpoints/parcel3d-002-8.0125.ckpt \
# MAGIC   --image /Volumes/playground/it_cl_terminal/ilo/data/blender_generated/images/img_0000.png \
# MAGIC   --conf 0.1 --out pred_0000.png