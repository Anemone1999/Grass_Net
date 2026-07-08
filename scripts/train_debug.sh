#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
cd ../
root_path=$(realpath ../)
# determine whether to run locally or to run remotely
if [[ "$root_path" == *"non_redundant_rotation_predict"* ]]; then
    echo "using relative path, running locally"
else
    echo "using remote path, running remotely"
    root_path="/mnt/castor-ice/workspace"
fi

rel_dataset_path="./datasets/ethanol/full_data"
index_path=$(realpath -m "${root_path}/${rel_dataset_path}")
dataset_path=$(realpath -m "${index_path}/data.mdb")

rel_hydra_path="./outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)"
hydra_path=$(realpath -m "${root_path}/${rel_hydra_path}")

python pipelines/train.py \
 hydra_path="${hydra_path}" \
 wandb.open=True \
 wandb.wandb_group="debug" \
 wandb.wandb_project="sphnet_debug" \
 wandb.wandb_name="debug_run" \
 data_name="md17_ethanol" \
 basis="def2-svp" \
 index_path="${index_path}" \
 dataset_path="${dataset_path}" \