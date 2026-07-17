#!/bin/bash
# Compare w=0.0 vs w=0.05 projection, 20 steps, per-step logging + grad_norm
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
  echo "=== w=${WEIGHT} (projection, 20 steps) ==="
  rm -rf "/tmp/compare_proj_w${WEIGHT}"

  export GRASS_DEBUG_GRAD=1
  export GRASS_DEBUG_JOB_ID="compare_w${WEIGHT}_proj"

  python pipelines/train.py \
    job_id=compare_proj_w${WEIGHT} \
    wandb.wandb_name=compare_proj_w${WEIGHT} \
    wandb.open=false wandb.wandb_project=grassD_train wandb.wandb_api_key=x \
    data_name=md17_ethanol "dataset_path=${DATASET}" "index_path=${INDEX}" \
    basis=def2-svp max_steps=20 num_epochs=3000000 batch_size=4 dataloader_num_workers=1 \
    hydra_path="/tmp/compare_proj_w${WEIGHT}" \
    log_every_n_steps=1 \
    skip_test=true \
    enable_hami=true \
    +enable_grassmann=true \
    +enable_stationarity=false \
    +grassmann_weight=${WEIGHT} \
    +grassmann_metric=projection \
    ed_type=naive ed_trunc_factor=3.0 gradient_clip_val=1.0 \
    seed=42 lr=0.001 hami_weight=1.0 \
    use_sparse_tp=false remove_init=true used_cache=false precision=32

  cp "/tmp/compare_proj_w${WEIGHT}/logs/log.txt" "${OUT_DIR}/log_proj_w${WEIGHT}_20steps.txt"

  # Append grad_norm from training log into the grad_debug log
  GRAD_DEBUG_LOG=$(ls -t "${OUT_DIR}/grad_debug_compare_w${WEIGHT}_proj_"*.log 2>/dev/null | head -1)
  if [ -n "$GRAD_DEBUG_LOG" ]; then
    python3 -c "
import re
step_norms = {}
cur_step = None
with open('${OUT_DIR}/log_proj_w${WEIGHT}_20steps.txt') as f:
    for line in f:
        m = re.search(r'step: ([\d.]+)', line)
        if m:
            cur_step = int(float(m.group(1)))
        g = re.search(r'train/grad_norm: ([\d.]+)', line)
        if g and cur_step is not None:
            norm = float(g.group(1))
            step_norms[cur_step] = norm
with open('$GRAD_DEBUG_LOG', 'a') as f:
    f.write('# grad_norm per step (from training log)\n')
    for step in sorted(step_norms):
        f.write(f'[step#{step}] grad_norm={step_norms[step]:.4f}\n')
print('Appended grad_norm to debug log')
"
  fi

  echo "Grad debug logs:"
  ls -la "${OUT_DIR}/grad_debug_compare_w${WEIGHT}_proj_"*.log 2>/dev/null || echo "  (none found)"
  echo ""
done

echo "=== Done ==="
ls -la ${OUT_DIR}/log_proj_w*.txt
