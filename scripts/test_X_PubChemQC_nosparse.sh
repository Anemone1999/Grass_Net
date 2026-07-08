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

model_name="DM1_EA10_nosparse"
rel_hydra_path="./outputs_main/X_pred/QH9/random/${model_name}"
hydra_path=$(realpath -m "${root_path}/${rel_hydra_path}")

heavy_atom_number_list=("10" "11" "12" "13" "14" "15" "16" "17")
# heavy_atom_number_list=("17")
for ((i=0; i<${#heavy_atom_number_list[@]}; i++)); do
    heavy_atom_number=${heavy_atom_number_list[i]}
    rel_dataset_path="./datasets/PubchemQC_${heavy_atom_number}/full_data_test"
    index_path=$(realpath -m "${root_path}/${rel_dataset_path}")
    dataset_path=$(realpath -m "${index_path}/data.mdb")
    rel_acc_path="./datasets/PubchemQC_${heavy_atom_number}/SCF_rounds"
    acc_path=$(realpath -m "${root_path}/${rel_acc_path}")
    log_path=${hydra_path}

    python pipelines/test.py \
    hydra_path="${hydra_path}" \
    precision="64" \
    wandb.open=True \
    wandb.wandb_group="debug" \
    wandb.wandb_project="SPHNet_test" \
    wandb.wandb_name="${model_name}_test" \
    data_name="PubchemQC_${heavy_atom_number}" \
    basis="def2-svp" \
    index_path="${index_path}" \
    dataset_path="${dataset_path}" \
    acc_path="${acc_path}" \
    pred_target="X" \
    enable_DM=True \
    remove_init=False \
    enable_exceed_pi=False \
    enable_ecore=False \
    enable_dipole=False \
    enable_total_energy=True \
    enable_acc_ratio=True \
    xc_type="b3lyp5" \
    use_sparse_tp=False \
    dataloader_num_workers=4 \
    max_sample_for_total_energy=100 \
    batch_size=1 \
    inference_batch_size=1 \
    2>&1 | tee >(grep --line-buffered -vE "it/s|v_num=|%\|" > \
    "${log_path}/OOD_NUM_${heavy_atom_number}.log")

done