#!/bin/bash
# Debug: 15 steps, per-step logging + per-chain grad norms
set -e
cd /home/pepe/codebench/GrassD_sphnet_ver2

export SWANLAB_MODE=disabled
export GRASS_DEBUG_GRAD=1

DATASET="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb/data.mdb"
INDEX="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp"
INDEX_FILE="MD17_ethanol_trainset_with_5000_data.pt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

ln -sf ${INDEX}/${INDEX_FILE} ${INDEX}/MD17_ethanol.pt

python pipelines/train.py \
  job_id=debug_grad_${TIMESTAMP} \
  wandb.wandb_name=debug_grad_${TIMESTAMP} \
  wandb.open=false \
  wandb.wandb_project=grassD_train \
  wandb.wandb_api_key=mVry0ZUvTErOCGFq1cBjV \
  data_name=md17_ethanol \
  "dataset_path=${DATASET}" \
  "index_path=${INDEX}" \
  basis=def2-svp \
  max_steps=15 \
  num_epochs=3000000 \
  batch_size=4 \
  inference_batch_size=4 \
  dataloader_num_workers=1 \
  hydra_path=/home/pepe/workbench/grassd_outputs/debug_grad_${TIMESTAMP} \
  +log_every_n_steps=1 \
  enable_hami=true \
  +enable_grassmann=true \
  +enable_stationarity=false \
  +grassmann_weight=0.05 \
  +grassmann_metric=geodesic \
  +grassmann_warmup_steps=0 \
  ed_type=naive \
  ed_trunc_factor=3.0 \
  gradient_clip_val=1.0 \
  seed=42 \
  lr=0.001 \
  hami_weight=1.0 \
  use_sparse_tp=false \
  remove_init=true \
  used_cache=false \
  precision=32
