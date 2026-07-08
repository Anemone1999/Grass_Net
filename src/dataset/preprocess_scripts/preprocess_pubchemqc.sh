#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
filename=$(basename "$0")

cd "$SCRIPT_DIR"
cd ../../../
root_path=$(realpath ../)
# determine whether to run locally or to run remotely
if [[ "$root_path" == *"castor-ice"* ]]; then
    echo "using relative path, running locally"
else
    echo "using remote path, running remotely"
    root_path="/mnt/castor-ice/workspace"
fi
heavy_atom_number_list=("10" "11" "12" "13" "14" "15" "16" "17")
for ((i=0; i<${#heavy_atom_number_list[@]}; i++)); do
    heavy_atom_number=${heavy_atom_number_list[i]}

    rel_dataset_input="./datasets"
    rel_dataset_output="./datasets/PubchemQC_${heavy_atom_number}/full_data_test"
    in_path=$(realpath -m "${root_path}/${rel_dataset_input}")
    out_path=$(realpath -m "${root_path}/${rel_dataset_output}")

    python src/dataset/preprocess_data.py \
    --data_name PubchemQC_${heavy_atom_number} \
    --input_path ${in_path} \
    --output_path ${out_path}
done