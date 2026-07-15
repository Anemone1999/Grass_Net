#!/bin/bash
set -e
cd /home/pepe/codebench/GrassD_sphnet_ver2

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="/home/pepe/workbench/grassd_outputs/svp_grassmann_test_${TIMESTAMP}"
mkdir -p "$OUTDIR"

DATASET="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb/data.mdb"
INDEX="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/MD17_ethanol.pt"

for MODEL_DIR in v2_projection_w0.05 v2_hami_control v2_grass_w1.0; do
    CKPT="/home/pepe/workbench/grassd_outputs/${MODEL_DIR}/best-epoch=0191-val_loss=0.000010.ckpt"
    if [ ! -f "$CKPT" ]; then
        CKPT=$(ls /home/pepe/workbench/grassd_outputs/${MODEL_DIR}/best*.ckpt 2>/dev/null | head -1)
    fi
    if [ ! -f "$CKPT" ]; then
        CKPT="/home/pepe/workbench/grassd_outputs/${MODEL_DIR}/last.ckpt"
    fi

    echo "=== $MODEL_DIR ==="
    python pipelines/test_grassd.py \
        --ckpt "$CKPT" \
        --dataset "$DATASET" \
        --index "$INDEX" \
        --basis def2-svp \
        --n_test 100 \
        --out "$OUTDIR" \
        --model_name "$MODEL_DIR"
done

echo "Done! Output: $OUTDIR"
