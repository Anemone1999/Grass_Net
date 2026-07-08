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

rel_dataset_input="../../data/nablaDFT/train/raw/dataset_train_medium.db"
rel_dataset_output="./datasets/NablaDFT_medium/full_data"
in_path=$(realpath -m "${root_path}/${rel_dataset_input}")
out_path=$(realpath -m "${root_path}/${rel_dataset_output}")

python src/dataset/preprocess_data.py \
 --data_name NablaDFT_medium \
 --input_path ${in_path} \
 --output_path ${out_path}