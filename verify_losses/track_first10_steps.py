"""Track grassmann_loss over first 10 training steps.
Simplified: one forward pass per step, compute loss, backward, update.
"""
import sys, os, re, pickle, lmdb, io
import numpy as np
import torch
import warnings
torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")

SRC = "/home/pepe/codebench/GrassD_sphnet_ver2/src"
sys.path.insert(0, os.path.join(SRC, os.pardir))
sys.path.insert(0, SRC)

from omegaconf import OmegaConf
from src.dataset.buildblock import get_conv_variable_lin
from src.dataset.utils import collate_fn_unified
from src.models.model import create_model

DEVICE = 'cpu'
DATASET = "/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb"
SPLIT = "/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/MD17_ethanol_trainset_with_5000_data.pt"

N_STEPS = 10
EPS_EIGH = 1e-8

# ====================================================
# Minimal grassmann computation (exact same formula as GrassmannError)
# ====================================================
def solve_eigh(H, S, nocc):
    s_vals, s_vecs = torch.linalg.eigh(S)
    s_vals = torch.clamp(s_vals, min=EPS_EIGH)
    X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
    Fs = X.T @ H @ X
    e_vals, e_vecs = torch.linalg.eigh(Fs)
    C = X @ e_vecs
    return C[:, :nocc]

def grassmann_geodesic(C_gt_occ, C_pred_occ, S):
    from src.training.losses import stable_acos
    M = C_gt_occ.T @ S @ C_pred_occ
    _, sigma, _ = torch.linalg.svd(M)
    theta = stable_acos(sigma.clamp(-1, 1))
    return (theta ** 2).sum()

def compute_hami_mae(H_pred, H_gt):
    return (H_pred - H_gt).abs().mean()

# ====================================================
# Build model from config
# ====================================================
print("Building model (seed=42) ...")
torch.manual_seed(42)
np.random.seed(42)
ckpt = torch.load("/home/pepe/workbench/grassd_outputs/v2_geodesic_w0.05_20260715_112607/logs/last.ckpt",
                  map_location='cpu')
hp = ckpt['hyper_parameters']
hp_dict = {}
for k in hp:
    try:
        v = hp[k]
        if isinstance(v, (list, tuple)): hp_dict[k] = list(v)
        elif hasattr(v, 'pretty'):
            try: hp_dict[k] = OmegaConf.to_container(v, resolve=True, throw_on_missing=False)
            except: hp_dict[k] = str(v)
        else: hp_dict[k] = v
    except: pass
model = create_model(hp_dict)
model.to(DEVICE).train()
print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

# ====================================================
# Data
# ====================================================
print("Loading data ...")
conv, _, mask, _ = get_conv_variable_lin('def2-svp')
collate = collate_fn_unified(long_cutoff_upper=9, unit=1)

env = lmdb.open(DATASET, readonly=True, lock=False, readahead=False, meminit=False, max_readers=32)
split = torch.load(SPLIT, map_location='cpu', weights_only=False)
test_ids = split[2].tolist()[:2]

import pyscf.gto
samples = []
with env.begin() as txn:
    for abs_idx in test_ids:
        raw = pickle.loads(txn.get(int(abs_idx).to_bytes(4, 'big')))
        atoms = np.frombuffer(raw['atoms'], dtype=np.int32)
        pos = np.frombuffer(raw['pos'], dtype=np.float64).reshape(len(atoms), 3)
        H_flat = np.frombuffer(raw['Ham'], dtype=np.float64)
        H_init = raw['Ham_init']
        norbs = H_init.shape[0]
        H = H_flat.reshape(norbs, norbs)
        mol = pyscf.gto.M(atom=[(int(z), pos[i]) for i,z in enumerate(atoms)],
                          basis='def2-svp', unit='ang', verbose=0)
        s1e = mol.intor('int1e_ovlp')
        samples.append({
            'atoms': atoms, 'pos': pos, 'H': H, 'H_init': H_init, 's1e': s1e,
            'norb': norbs, 'nocc': int(np.sum(atoms)) // 2,
            'labels': raw['labels'], 'edge_index': raw['edge_index'],
            'num_nodes': raw['num_nodes'],
        })
env.close()

entries = []
for i, s in enumerate(samples):
    entries.append({
        'idx': i, 'cid': i,
        'pos': s['pos'], 'atomic_numbers': s['atoms'],
        'molecule_size': len(s['pos']),
        'fock': s['H'] - s['H_init'],
        'fock_init': s['H_init'].copy(),
        'C_init': np.eye(s['norb']),
        'D_gt': np.eye(s['norb']),
        'overlap': s['s1e'],
        'n_orb': 0,
        'labels': s['labels'],
        'edge_index': np.asarray(s['edge_index']),
        'buildblock_mask': mask, 'max_block_size': conv.max_block_size,
        'grid_coords': np.array([0], dtype=np.float64),
        'grid_weights': np.array([0], dtype=np.float64),
        'num_nodes': s['num_nodes'],
    })

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\n{'='*75}")
print(f"{'Step':>5s}  {'hami_mae':>12s}  {'geodesic':>12s}  {'geo(log10)':>10s}")
print(f"{'-'*75}")

for step in range(N_STEPS + 1):
    # Fresh collate each step to avoid mutated batch
    batch = collate([dict(e) for e in entries]).to(DEVICE)

    out = model(batch)
    if 'pred_hamiltonian_diagonal_blocks' not in out:
        out = model.hami_model(out)
    out = model.hami_model.build_final_matrix(out, sym_type='sym')

    # Check if output requires grad
    if step == 1:
        has_grad = out['pred_hamiltonian_diagonal_blocks'].requires_grad
        print(f"[DEBUG step1] pred_ham requires_grad={has_grad}")

    # Add back fock_init
    pred_hams = []
    gt_hams = []
    for i, s in enumerate(samples):
        fi = torch.tensor(s['H_init'], dtype=torch.float64, device=DEVICE)
        pred_hams.append(out['pred_hamiltonian'][i] + fi)
        gt_hams.append(torch.tensor(s['H'], dtype=torch.float64, device=DEVICE))

    # Compute losses
    hami_losses = [compute_hami_mae(p, g) for p, g in zip(pred_hams, gt_hams)]
    hami_mean = torch.stack(hami_losses).mean()

    geo_losses = []
    for i, s in enumerate(samples):
        S = torch.tensor(s['s1e'], dtype=torch.float64, device=DEVICE)
        C_gt_occ = solve_eigh(gt_hams[i], S, s['nocc'])
        C_pred_occ = solve_eigh(pred_hams[i], S, s['nocc'])
        geo_losses.append(grassmann_geodesic(C_gt_occ, C_pred_occ, S))
    geo_mean = torch.stack(geo_losses).mean()

    total_loss = hami_mean + 0.05 * geo_mean

    # Log
    geo_val = geo_mean.detach().item()
    geo_log = np.log10(geo_val) if geo_val > 0 else float('-inf')
    print(f"{step:5d}  {hami_mean.item():12.6e}  {geo_val:12.6f}  {geo_log:10.4f}")

    if step >= N_STEPS:
        break

    # Gradient step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

print(f"{'='*75}")
print("Done.")
