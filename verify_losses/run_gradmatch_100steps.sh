#!/bin/bash
# Test EMA-based gradient matching: w=0.05 geodesic, 100 steps
set -e
cd /home/pepe/codebench/GrassD_sphnet_ver2

export SWANLAB_MODE=disabled
OUT_DIR=/home/pepe/codebench/GrassD_sphnet_ver2/verify_losses
mkdir -p $OUT_DIR

DATASET="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb/data.mdb"
INDEX="/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp"
INDEX_FILE="MD17_ethanol_trainset_with_5000_data.pt"
ln -sf ${INDEX}/${INDEX_FILE} ${INDEX}/MD17_ethanol.pt

RUN_NAME="gradmatch_w0.05"
echo "=== ${RUN_NAME} (geodesic, 100 steps) ==="
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
  ed_type=naive ed_trunc_factor=3.0 gradient_clip_val=1.0 \
  seed=42 lr=0.001 hami_weight=1.0 \
  use_sparse_tp=false remove_init=true used_cache=false precision=32

cp "/tmp/${RUN_NAME}/logs/log.txt" "${OUT_DIR}/log_${RUN_NAME}_100steps.txt"

echo "=== Extracting summary ==="
python3 -c "
import re

cur_step = None
out = open('${OUT_DIR}/${RUN_NAME}_summary.txt', 'w')
out.write('step\tloss\thami_mae\tgrassmann_loss\tgrad_norm\n')
count = 0
with open('${OUT_DIR}/log_${RUN_NAME}_100steps.txt') as f:
    for line in f:
        m = re.search(r'step: ([\d.]+)', line)
        if m:
            cur_step = int(float(m.group(1)))
        g = re.search(r'train/grad_norm: ([\d.]+)', line)
        if g and cur_step is not None:
            loss = re.search(r'train_per_step/loss: ([\d.eE+-]+)', line)
            hami = re.search(r'train_per_step/hami_loss_mae: ([\d.eE+-]+)', line)
            grass = re.search(r'train_per_step/grassmann_loss: ([\d.eE+-]+)', line)
            gn = float(g.group(1))
            ls = float(loss.group(1)) if loss else -1.0
            hm = float(hami.group(1)) if hami else -1.0
            gr = float(grass.group(1)) if grass else -1.0
            # Dedup: only write first occurrence per step
            if cur_step not in seen:
                seen.add(cur_step)
                out.write(f'{cur_step}\t{ls:.6f}\t{hm:.6f}\t{gr:.6f}\t{gn:.6f}\n')
                count += 1
out.close()

print(f'Wrote {count} steps')
print('First 5:')
" 
# The seen set needs to be initialized
python3 -c "
import re
seen = set()
cur_step = None
count = 0
with open('${OUT_DIR}/log_${RUN_NAME}_100steps.txt') as f:
    for line in f:
        m = re.search(r'step: ([\d.]+)', line)
        if m: cur_step = int(float(m.group(1)))
        g = re.search(r'train/grad_norm: ([\d.]+)', line)
        if g and cur_step is not None and cur_step not in seen:
            seen.add(cur_step)
            loss = re.search(r'train_per_step/loss: ([\d.eE+-]+)', line)
            hami = re.search(r'train_per_step/hami_loss_mae: ([\d.eE+-]+)', line)
            grass = re.search(r'train_per_step/grassmann_loss: ([\d.eE+-]+)', line)
            ls = float(loss.group(1)) if loss else -1.0
            hm = float(hami.group(1)) if hami else -1.0
            gr = float(grass.group(1)) if grass else -1.0
            gn = float(g.group(1))
            if count == 0:
                with open('${OUT_DIR}/${RUN_NAME}_summary.txt', 'w') as out:
                    out.write(f'step\tloss\thami_mae\tgrassmann_loss\tgrad_norm\n')
            with open('${OUT_DIR}/${RUN_NAME}_summary.txt', 'a') as out:
                out.write(f'{cur_step}\t{ls:.6f}\t{hm:.6f}\t{gr:.6f}\t{gn:.6f}\n')
            count += 1
print(f'Wrote {count} steps to ${OUT_DIR}/${RUN_NAME}_summary.txt')
print()
with open('${OUT_DIR}/${RUN_NAME}_summary.txt') as f:
    lines = f.readlines()
    for line in lines[:6]:
        print(line.strip())
    print('...')
    for line in lines[-5:]:
        print(line.strip())
"

echo ""
echo "=== Done ==="