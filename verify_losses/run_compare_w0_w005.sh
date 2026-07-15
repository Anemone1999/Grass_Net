#!/bin/bash
# Compare w=0.0 vs w=0.05 geodesic, 20 steps, per-step logging, skip test
set -e
cd /home/pepe/codebench/GrassD_sphnet_ver2

export SWANLAB_MODE=disabled
OUT_DIR=/home/pepe/codebench/GrassD_sphnet_ver2/verify_losses
mkdir -p $OUT_DIR

DATASET="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb/data.mdb"
INDEX="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp"
INDEX_FILE="MD17_ethanol_trainset_with_5000_data.pt"
ln -sf ${INDEX}/${INDEX_FILE} ${INDEX}/MD17_ethanol.pt

for WEIGHT in 0.0 0.05; do
  echo "=== w=${WEIGHT} (geodesic, 20 steps) ==="
  rm -rf "/tmp/compare_w${WEIGHT}"

  # + prefix required for keys not in config.yaml or hydra_config.py
  python pipelines/train.py \
    job_id=compare_w${WEIGHT} \
    wandb.wandb_name=compare_w${WEIGHT} \
    wandb.open=false wandb.wandb_project=grassD_train wandb.wandb_api_key=x \
    data_name=md17_ethanol "dataset_path=${DATASET}" "index_path=${INDEX}" \
    basis=def2-svp max_steps=20 num_epochs=3000000 batch_size=4 dataloader_num_workers=1 \
    hydra_path="/tmp/compare_w${WEIGHT}" \
    log_every_n_steps=1 \
    skip_test=true \
    enable_hami=true \
    +enable_grassmann=true \
    +enable_stationarity=false \
    +grassmann_weight=${WEIGHT} \
    +grassmann_metric=geodesic \
    +grassmann_warmup_steps=0 \
    ed_type=naive ed_trunc_factor=3.0 gradient_clip_val=1.0 \
    seed=42 lr=0.001 hami_weight=1.0 \
    use_sparse_tp=false remove_init=true used_cache=false precision=32

  cp "/tmp/compare_w${WEIGHT}/logs/log.txt" "${OUT_DIR}/log_w${WEIGHT}_20steps.txt"
  echo ""
done

echo "=== Done ==="
ls -la ${OUT_DIR}/log_w*.txt
