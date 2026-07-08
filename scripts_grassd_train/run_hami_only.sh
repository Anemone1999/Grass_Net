#!/bin/bash
# Task 1: hami*1 only (Hamiltonian loss only, no grassmann)
set -e

DATASET="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb/data.mdb"
INDEX="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp"
INDEX_FILE="MD17_ethanol_trainset_with_5000_data.pt"

cd /home/pepe/codebench/GrassD_sphnet_ver2

# Symlink the specific split file so data.py correctly picks it up
ln -sf ${INDEX}/${INDEX_FILE} ${INDEX}/MD17_ethanol.pt

python pipelines/train.py \
  data_name=md17_ethanol \
  "dataset_path=${DATASET}" \
  "index_path=${INDEX}" \
  basis=def2-svp \
  max_steps=60000 \
  num_epochs=3000000 \
  batch_size=16 \
  inference_batch_size=16 \
  dataloader_num_workers=8 \
  enable_hami=true \
  +enable_grassmann=false \
  +enable_stationarity=false \
  +grassmann_weight=0.0 \
  ed_type=naive \
  ed_trunc_factor=3.0 \
  gradient_clip_val=0.16 \
  seed=42 \
  lr=0.001 \
  hami_weight=1.0 \
  use_sparse_tp=false \
  remove_init=true \
  used_cache=false \
  job_id=v2_hami_only \
  wandb.open=true \
  wandb.wandb_name=v2_hami_only \
  wandb.wandb_project=grassD_train \
  wandb.wandb_api_key=mVry0ZUvTErOCGFq1cBjV \
  precision=32
