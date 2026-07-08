#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
filename=$(basename "$0")
wandb_name="${filename%.sh}"

cd "$SCRIPT_DIR"
cd ../
root_path=$(realpath ../)
# determine whether to run locally or to run remotely
if [[ "$root_path" == *"castor-ice"* ]]; then
    echo "using relative path, running locally"
else
    echo "using remote path, running remotely"
    root_path="/mnt/castor-ice/workspace"
fi

N=500
index_path="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_ethanol/def2qzvp/index_${N}"
dataset_path="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_ethanol/def2qzvp/data.mdb/data.mdb"

OUTPUT_BASE="/home/pepe/workbench/basisset_scaling/spnethami_output/def2qzvp_${N}_$(date +%Y%m%d)"
hydra_path="${OUTPUT_BASE}/${wandb_name}"
log_path="${OUTPUT_BASE}/logs"
mkdir -p "${hydra_path}" "${log_path}"


python pipelines/train.py \
 hydra_path="${hydra_path}" \
 precision="32" \
 wandb.open=True \
 wandb.wandb_group="debug" \
 wandb.wandb_project="sphnet_debug" \
 wandb.wandb_name=${wandb_name} \
 dataloader_num_workers=8 \
 batch_size=4 \
 inference_batch_size=4 \
 data_name="md17_ethanol" \
 xc_type="pbe" \
 basis="def2-qzvp" \
 model.order=8 \
 index_path="${index_path}" \
 dataset_path="${dataset_path}" \
 pred_target="H" \
 enable_hami=True \
 hami_weight=1 \
 remove_init=True \
 sparsity=0.7 \
 max_steps=200000 \
 2>&1 | tee >(grep --line-buffered -vE "it/s|v_num=|%\|" > \
 "${log_path}/$(date +%Y-%m-%d)-$(date +%H-%M-%S)-${wandb_name}.log")
