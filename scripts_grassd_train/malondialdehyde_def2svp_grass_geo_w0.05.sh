#!/bin/bash
# SPHNet: malondialdehyde def2-svp + grassmann_loss (geodesic, w=0.05)
SCRIPT_DIR=$(dirname "$(realpath "$0")")
wandb_name="malondialdehyde_def2svp_grass_geo_w0.05"

cd "$SCRIPT_DIR"
cd ../

INDEX_PATH="/home/pepe/workbench/basisset_scaling/lmdb_data/gcnet_malondialdehyde/def2-svp"
DATASET_PATH="/home/pepe/workbench/basisset_scaling/lmdb_data/gcnet_malondialdehyde/def2-svp/data.mdb/data.mdb"
HYDRA_PATH="/home/pepe/workbench/basisset_scaling/sphnet_output/def2svp_grass/${wandb_name}"

python pipelines/train.py \
 hydra_path="${HYDRA_PATH}" \
 precision="32" \
 wandb.open=True \
 dataloader_num_workers=12 \
 batch_size=32 \
 inference_batch_size=32 \
 data_name="md17_malondialdehyde" \
 xc_type="pbe" \
 basis="def2-svp" \
 model.order=4 \
 index_path="${INDEX_PATH}" \
 dataset_path="${DATASET_PATH}" \
 pred_target='H' \
 remove_init=True \
 +check_val_every_n_epoch=2 \
 use_sparse_tp=False \
 gradient_clip_val=0.09 \
 wandb.wandb_project="SPHNet" \
 wandb.wandb_group="malondialdehyde_grassmann" \
 wandb.wandb_name=${wandb_name} \
 max_steps=200000 \
 seed=10086 \
 +enable_grassmann=true \
 +enable_stationarity=false \
 +grassmann_metric=geodesic \
 +grassmann_weight=0.05
