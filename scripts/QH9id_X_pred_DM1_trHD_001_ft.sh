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

rel_dataset_path="./datasets/QH9Stable/full_data"
index_path=$(realpath -m "${root_path}/${rel_dataset_path}")
dataset_path=$(realpath -m "${index_path}/data.mdb")

rel_hydra_path="./outputs_sphnet/$(date +%Y-%m-%d)/$(date +%H-%M-%S)/${wandb_name}"
hydra_path=$(realpath -m "${root_path}/${rel_hydra_path}")

rel_log_path="./log_sphnet/train"
log_path=$(realpath -m "${root_path}/${rel_log_path}")

rel_finetune_path="./outputs_main/X_pred/QH9/random/DM1_EA0/logs"
finetune_path=$(realpath -m "${root_path}/${rel_finetune_path}")
rel_finetune_ckpt_path="./outputs_main/X_pred/QH9/random/DM1_EA0/ckpts"
finetune_ckpt_path=$(realpath -m "${root_path}/${rel_finetune_ckpt_path}")


python pipelines/train.py \
 hydra_path="${hydra_path}" \
 finetune_flag=True \
 finetune_path="${finetune_path}" \
 finetune_ckpt_path="${finetune_ckpt_path}" \
 max_steps=100000 \
 lr=1e-4 \
 precision="32" \
 dataloader_num_workers=4 \
 batch_size=32 \
 inference_batch_size=32 \
 wandb.open=True \
 wandb.wandb_group="debug" \
 wandb.wandb_project="sphnet_debug" \
 wandb.wandb_name=${wandb_name} \
 data_name="qh9_stable_iid" \
 basis="def2-svp" \
 index_path="${index_path}" \
 dataset_path="${dataset_path}" \
 pred_target="X" \
 enable_DM=True \
 dm_weight=1 \
 remove_init=False \
 enable_exceed_pi=False \
 enable_trHD=True \
 trHD_weight=0.01 \
 sparsity=0.4 \
 2>&1 | tee >(grep --line-buffered -vE "it/s|v_num=|%\|" > \
 "${log_path}/$(date +%Y-%m-%d)-$(date +%H-%M-%S)-${wandb_name}.log")