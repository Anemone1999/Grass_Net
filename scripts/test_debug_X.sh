#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
wandb_name="${filename%.sh}"
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

model_name="SPHNet_X_pred_EA_ft"
rel_hydra_path="./outputs_models/${model_name}"
hydra_path=$(realpath -m "${root_path}/${rel_hydra_path}")


CUDA_VISIBLE_DEVICES=0 python pipelines/test.py \
 hydra_path="${hydra_path}" \
 precision="64" \
 wandb.open=True \
 wandb.wandb_group="debug" \
 wandb.wandb_project="SPHNet_test" \
 wandb.wandb_name="${model_name}_test" \
 data_name="md17_ethanol" \
 basis="def2-svp" \
 index_path="${index_path}" \
 dataset_path="${dataset_path}" \
 pred_target="X" \
 enable_DM=True \
 remove_init=False \
 enable_exceed_pi=False \
 enable_total_energy=True \
 xc_type="pbe" \
 max_sample_for_total_energy=100 \
 batch_size=1 \
 inference_batch_size=1