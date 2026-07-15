#!/bin/bash
# Task: hami*1 + grassd_loss(geodesic)*1.0
set -e

DATASET="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb/data.mdb"
INDEX="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp"
INDEX_FILE="MD17_ethanol_trainset_with_5000_data.pt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TASK_NAME="v2_geodesic_w1.0_${TIMESTAMP}"

cd /home/pepe/codebench/GrassD_sphnet_ver2

ln -sf ${INDEX}/${INDEX_FILE} ${INDEX}/MD17_ethanol.pt

python pipelines/train.py \
  job_id=${TASK_NAME} \
  wandb.wandb_name=${TASK_NAME} \
  wandb.open=true \
  wandb.wandb_project=grassD_train \
  wandb.wandb_api_key=mVry0ZUvTErOCGFq1cBjV \
  data_name=md17_ethanol \
  "dataset_path=${DATASET}" \
  "index_path=${INDEX}" \
  basis=def2-svp \
  max_steps=60000 \
  num_epochs=3000000 \
  batch_size=16 \
  inference_batch_size=16 \
  dataloader_num_workers=8 \
  hydra_path=/home/pepe/workbench/grassd_outputs/${TASK_NAME} \
  enable_hami=true \
  +enable_grassmann=true \
  +enable_stationarity=false \
  +grassmann_weight=1.0 \
  +grassmann_metric=geodesic \
  ed_type=naive \
  ed_trunc_factor=3.0 \
  gradient_clip_val=0.16 \
  seed=42 \
  lr=0.001 \
  hami_weight=1.0 \
  use_sparse_tp=false \
  remove_init=true \
  used_cache=false \
  precision=32
