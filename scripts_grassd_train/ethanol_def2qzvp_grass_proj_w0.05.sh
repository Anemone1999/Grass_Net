#!/bin/bash
# SPHNet: ethanol def2-qzvp + grassmann_loss (projection, w=0.05)
SCRIPT_DIR=$(dirname "$(realpath "$0")")
wandb_name="ethanol_def2qzvp_grass_proj_w0.05_clamp"

cd "$SCRIPT_DIR"
cd ../

INDEX_PATH="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2qzvp/index_full"
DATASET_PATH="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2qzvp/data.mdb/data.mdb"
HYDRA_PATH="/home/pepe/workbench/basisset_scaling/sphnet_output/def2qzvp_grass/${wandb_name}"

python pipelines/train.py \
 hydra_path="${HYDRA_PATH}" \
 precision="32" \
 wandb.open=True \
 dataloader_num_workers=12 \
 batch_size=32 \
 inference_batch_size=32 \
 data_name="md17_ethanol" \
 xc_type="pbe" \
 basis="def2-qzvp" \
 model.order=8 \
 index_path="${INDEX_PATH}" \
 dataset_path="${DATASET_PATH}" \
 pred_target='H' \
 remove_init=True \
 +check_val_every_n_epoch=2 \
 use_sparse_tp=False \
 gradient_clip_val=0.09 \
 wandb.wandb_project="SPHNet" \
 wandb.wandb_group="ethanol_grassmann" \
 wandb.wandb_name=${wandb_name} \
 max_steps=200000 \
 seed=10086 \
 +enable_grassmann=true \
 +enable_stationarity=false \
 +grassmann_metric=projection \
 +grassmann_weight=0.05
