#!/bin/bash
# QH9: H prediction + geodesic grassmann loss
SCRIPT_DIR=$(dirname "$(realpath "$0")")
wandb_name="qh9_geodesic_grassmann"

cd "$SCRIPT_DIR"
cd ../

INDEX_PATH="/home/pepe/data/QH9Stable/full_data"
DATASET_PATH="/home/pepe/data/QH9Stable/full_data/data.mdb"
HYDRA_PATH="/home/pepe/workbench/grassd_outputs/${wandb_name}"

python pipelines/train.py \
 hydra_path="${HYDRA_PATH}" \
 precision="32" \
 dataloader_num_workers=4 \
 batch_size=32 \
 inference_batch_size=32 \
 wandb.open=True \
 wandb.wandb_group="grassmann_qh9" \
 wandb.wandb_project="grassD_train" \
 wandb.wandb_name=${wandb_name} \
 data_name="qh9_stable_iid" \
 basis="def2-svp" \
 index_path="${INDEX_PATH}" \
 dataset_path="${DATASET_PATH}" \
 remove_init=True \
 +check_val_every_n_epoch=2 \
 pred_target='H' \
 use_sparse_tp=False \
 gradient_clip_val=0.09 \
 +enable_grassmann=true \
 +enable_stationarity=false \
 +grassmann_metric=geodesic \
 +grassmann_weight=0.05
