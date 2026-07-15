#!/bin/bash
# Quick test: 2 steps, w=0.05 geodesic, debug grass_loss
set -e
cd /home/pepe/codebench/GrassD_sphnet_ver2

export SWANLAB_MODE=disabled
find . -path "*/__pycache__/*" -delete 2>/dev/null
rm -rf /tmp/debug_final

python pipelines/train.py \
  job_id=debug_final_test \
  wandb.wandb_name=debug_final_test wandb.open=false \
  wandb.wandb_project=grassD_train wandb.wandb_api_key=x \
  data_name=md17_ethanol \
  dataset_path="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb/data.mdb" \
  index_path="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp" \
  basis=def2-svp max_steps=2 num_epochs=3000000 batch_size=4 dataloader_num_workers=1 \
  hydra_path=/tmp/debug_final log_every_n_steps=1 \
  enable_hami=true enable_grassmann=true enable_stationarity=false \
  grassmann_weight=0.05 grassmann_metric=geodesic grassmann_warmup_steps=0 \
  ed_type=naive ed_trunc_factor=3.0 gradient_clip_val=1.0 \
  seed=42 lr=0.001 hami_weight=1.0 \
  use_sparse_tp=false remove_init=true used_cache=false precision=32
