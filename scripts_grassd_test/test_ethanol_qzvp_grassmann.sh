#!/bin/bash
set -e
cd /home/pepe/codebench/GrassD_sphnet_ver2

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="/home/pepe/workbench/grassd_outputs/qzvp_ethanol_test/grassmann_${TIMESTAMP}"
mkdir -p "$OUTDIR"

DATASET="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2qzvp/data.mdb/data.mdb"
INDEX="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2qzvp/index_2000/MD17_ethanol.pt"

for TRAIN_SIZE in 2000 100; do
    CKPT="/home/pepe/workbench/basisset_scaling/sphnet_output/def2qzvp_${TRAIN_SIZE}/ethanol_${TRAIN_SIZE}_def2qzvp_sphnet/logs/best.ckpt"

    echo "=== Testing ${TRAIN_SIZE}trainset ==="
    python pipelines/test_grassd.py \
        --ckpt "$CKPT" \
        --dataset "$DATASET" \
        --index "$INDEX" \
        --basis def2-qzvp \
        --n_test 25 \
        --out "$OUTDIR" \
        --model_name "${TRAIN_SIZE}trainset"
done

echo "Done! Output: $OUTDIR"
