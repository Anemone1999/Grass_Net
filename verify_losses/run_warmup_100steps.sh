#!/bin/bash
# Test warming: 50-step warmup + 50-step full weight
set -e
cd /home/pepe/codebench/GrassD_sphnet_ver2

export SWANLAB_MODE=disabled
OUT_DIR=/home/pepe/codebench/GrassD_sphnet_ver2/verify_losses
mkdir -p $OUT_DIR

DATASET="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb/data.mdb"
INDEX="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp"
INDEX_FILE="MD17_ethanol_trainset_with_5000_data.pt"
ln -sf ${INDEX}/${INDEX_FILE} ${INDEX}/MD17_ethanol.pt

RUN_NAME="warmup_w0.05"
echo "=== ${RUN_NAME} (geodesic, 100 steps, 50-step warmup) ==="
rm -rf "/tmp/${RUN_NAME}"

python pipelines/train.py \
  job_id=${RUN_NAME} \
  wandb.wandb_name=${RUN_NAME} wandb.open=false wandb.wandb_project=grassD_train wandb.wandb_api_key=x \
  data_name=md17_ethanol "dataset_path=${DATASET}" "index_path=${INDEX}" \
  basis=def2-svp max_steps=100 num_epochs=3000000 batch_size=4 dataloader_num_workers=1 \
  hydra_path="/tmp/${RUN_NAME}" \
  log_every_n_steps=1 \
  skip_test=true \
  enable_hami=true \
  +enable_grassmann=true \
  +enable_stationarity=false \
  +grassmann_weight=0.05 \
  +grassmann_metric=geodesic \
  +grassmann_warmup_steps=50 \
  ed_type=naive ed_trunc_factor=3.0 gradient_clip_val=1.0 \
  seed=42 lr=0.001 hami_weight=1.0 \
  use_sparse_tp=false remove_init=true used_cache=false precision=32

cp "/tmp/${RUN_NAME}/logs/log.txt" "${OUT_DIR}/log_${RUN_NAME}_100steps.txt"

echo "=== Extracting summary ==="
python3 << 'PYEOF'
import re
log_path = f'/home/pepe/codebench/GrassD_sphnet_ver2/verify_losses/log_warmup_w0.05_100steps.txt'
summary_path = f'/home/pepe/codebench/GrassD_sphnet_ver2/verify_losses/warmup_w0.05_summary.txt'

seen = set()
cur_step = None
count = 0
with open(summary_path, 'w') as out:
    out.write('step\tloss\thami_mae\tgrassmann_loss\tgrad_norm\n')
    with open(log_path) as f:
        for line in f:
            m = re.search(r'step: ([\d.]+)', line)
            if m: cur_step = int(float(m.group(1)))
            g = re.search(r'train/grad_norm: ([\d.]+)', line)
            if g and cur_step is not None and cur_step not in seen:
                seen.add(cur_step)
                loss = re.search(r'train_per_step/loss: ([\d.eE+-]+)', line)
                hami = re.search(r'train_per_step/hami_loss_mae: ([\d.eE+-]+)', line)
                grass = re.search(r'train_per_step/grassmann_loss: ([\d.eE+-]+)', line)
                out.write(
                    f'{cur_step}\t'
                    f'{float(loss.group(1)) if loss else -1.0:.6f}\t'
                    f'{float(hami.group(1)) if hami else -1.0:.6f}\t'
                    f'{float(grass.group(1)) if grass else -1.0:.6f}\t'
                    f'{float(g.group(1)):.6f}\n'
                )
                count += 1

with open(summary_path) as f:
    lines = f.readlines()
    print(f'Wrote {count} steps')
    print()
    for line in lines[:6]:
        print(line.strip())
    print('...')
    for line in lines[-5:]:
        print(line.strip())
PYEOF

echo ""
echo "=== Done ==="
