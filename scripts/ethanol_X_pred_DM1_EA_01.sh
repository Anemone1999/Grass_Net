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

rel_dataset_path="./datasets/ethanol/full_data"
index_path=$(realpath -m "${root_path}/${rel_dataset_path}")
dataset_path=$(realpath -m "${index_path}/data.mdb")

rel_hydra_path="./outputs_sphnet/$(date +%Y-%m-%d)/$(date +%H-%M-%S)/${wandb_name}"
hydra_path=$(realpath -m "${root_path}/${rel_hydra_path}")

rel_log_path="./log_sphnet/train"
log_path=$(realpath -m "${root_path}/${rel_log_path}")


python pipelines/train.py \
 hydra_path="${hydra_path}" \
 precision="32" \
 wandb.open=True \
 wandb.wandb_group="debug" \
 wandb.wandb_project="sphnet_debug" \
 wandb.wandb_name=${wandb_name} \
 data_name="md17_ethanol" \
 basis="def2-svp" \
 index_path="${index_path}" \
 dataset_path="${dataset_path}" \
 pred_target="X" \
 enable_DM=True \
 dm_weight=1 \
 remove_init=False \
 enable_exceed_pi=False \
 enable_energy_align_DM=True \
 energy_align_dm_weight=0.1 \
 | tee "${log_path}/$(date +%Y-%m-%d)-$(date +%H-%M-%S)-${wandb_name}.log"