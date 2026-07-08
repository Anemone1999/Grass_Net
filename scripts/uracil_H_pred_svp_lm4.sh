#!/bin/bash
# SPHNet-Hami: uracil def2-svp order=4 max_steps=200000 batch_size=32 (8:1:1 split)
SCRIPT_DIR=$(dirname "$(realpath "$0")")
filename=$(basename "$0")
wandb_name="${filename%.sh}"

cd "$SCRIPT_DIR"
cd ../

DATASET_PATH="/home/pepe/workbench/basisset_scaling/lmdb_data/gcnet_uracil/def2-svp/data.mdb/data.mdb"
INDEX_PATH="/home/pepe/workbench/basisset_scaling/lmdb_data/gcnet_uracil/def2-svp"

OUTPUT_BASE="/home/pepe/workbench/basisset_scaling/spnethami_output/uracil_svp_$(date +%Y%m%d)"
hydra_path="${OUTPUT_BASE}/${wandb_name}"
log_path="${OUTPUT_BASE}/logs"
mkdir -p "${hydra_path}" "${log_path}"

python pipelines/train.py \
 hydra_path="${hydra_path}" \
 precision="32" \
 wandb.open=True \
 wandb.wandb_group="debug" \
 wandb.wandb_project="sphnet_debug" \
 wandb.wandb_name="${wandb_name}" \
 dataloader_num_workers=8 \
 batch_size=32 \
 inference_batch_size=32 \
 data_name="md17_uracil" \
 xc_type="pbe" \
 basis="def2-svp" \
 model.order=4 \
 index_path="${INDEX_PATH}" \
 dataset_path="${DATASET_PATH}" \
 pred_target="H" \
 enable_hami=True \
 hami_weight=1 \
 remove_init=True \
 sparsity=0.7 \
 max_steps=200000 \
 2>&1 | tee >(grep --line-buffered -vE "it/s|v_num=|%\|" > \
 "${log_path}/$(date +%Y-%m-%d)-$(date +%H-%M-%S)-${wandb_name}.log")
