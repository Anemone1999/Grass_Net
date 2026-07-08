#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
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

rel_dataset_path="./datasets/ethanol/full_data_def2tzvp"
index_path=$(realpath -m "${root_path}/${rel_dataset_path}")
dataset_path=$(realpath -m "${index_path}/data.mdb/data.mdb")

rel_acc_path="./datasets/ethanol/SCF_rounds_tzvp"
acc_path=$(realpath -m "${root_path}/${rel_acc_path}")

model_name="DM1_EA1_tzvp_lm6"
rel_hydra_path="./outputs_main/X_pred/ethanol/basis_scaling/${model_name}"
hydra_path=$(realpath -m "${root_path}/${rel_hydra_path}")

python pipelines/test.py \
 hydra_path="${hydra_path}" \
 precision="64" \
 wandb.open=True \
 wandb.wandb_group="debug" \
 wandb.wandb_project="SPHNet_test" \
 wandb.wandb_name="${model_name}_test" \
 data_name="md17_ethanol" \
 basis="def2-tzvp" \
 model.order=6 \
 index_path="${index_path}" \
 dataset_path="${dataset_path}" \
 acc_path="${acc_path}" \
 pred_target="X" \
 enable_DM=True \
 remove_init=False \
 enable_exceed_pi=False \
 enable_ecore=False \
 enable_dipole=False \
 enable_trHD=False \
 trHD_val_loss="rmaenmae" \
 enable_total_energy=True \
 enable_acc_ratio=True \
 enable_DM_based_H=True \
 enable_rho=False \
 xc_type="pbe" \
 sparsity=0.7 \
 dataloader_num_workers=4 \
 max_sample_for_total_energy=50 \
 batch_size=1 \
 inference_batch_size=1