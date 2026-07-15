"""Diagnose: track grad_norm at each step of the grassmann loss chain.

Chain: H_pred → eigh(S) → C_occ → M_cross → SVD → acos → θ² → sum
Checks: total grad, eigh contribution, SVD contribution, acos contribution.
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
from src.training.losses import stable_acos, StableEigh

DEVICE = 'cpu'
DATASET = "/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb"
SPLIT = "/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/MD17_ethanol_trainset_with_5000_data.pt"
EPS_EIGH = 1e-8

# ====================================================
# Load model + data
# ====================================================
print("Loading model (seed=42) + data ...")
torch.manual_seed(42); np.random.seed(42)

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
model = create_model(hp_dict).to(DEVICE).train()

conv, _, mask, _ = get_conv_variable_lin('def2-svp')
collate = collate_fn_unified(long_cutoff_upper=9, unit=1)
env = lmdb.open(DATASET, readonly=True, lock=False, readahead=False, meminit=False, max_readers=32)
split = torch.load(SPLIT, map_location='cpu', weights_only=False)
test_ids = split[2].tolist()[:4]

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
        samples.append({'atoms': atoms, 'pos': pos, 'H': H, 'H_init': H_init, 's1e': s1e,
                        'norb': norbs, 'nocc': int(np.sum(atoms))//2,
                        'labels': raw['labels'], 'edge_index': raw['edge_index'],
                        'num_nodes': raw['num_nodes']})
env.close()

entries = []
for i, s in enumerate(samples):
    entries.append({
        'idx': i, 'cid': i, 'pos': s['pos'], 'atomic_numbers': s['atoms'],
        'molecule_size': len(s['pos']), 'fock': s['H'] - s['H_init'],
        'fock_init': s['H_init'].copy(), 'C_init': np.eye(s['norb']),
        'D_gt': np.eye(s['norb']), 'overlap': s['s1e'], 'n_orb': 0,
        'labels': s['labels'], 'edge_index': np.asarray(s['edge_index']),
        'buildblock_mask': mask, 'max_block_size': conv.max_block_size,
        'grid_coords': np.array([0], dtype=np.float64),
        'grid_weights': np.array([0], dtype=np.float64),
        'num_nodes': s['num_nodes'],
    })

batch = collate([dict(e) for e in entries]).to(DEVICE)
out = model(batch)
out = model.hami_model(out)
out = model.hami_model.build_final_matrix(out, sym_type='sym')

# Add back fock_init
for i, s in enumerate(samples):
    fi = torch.tensor(s['H_init'], dtype=torch.float64, device=DEVICE)
    out['pred_hamiltonian'][i] = out['pred_hamiltonian'][i] + fi
    out['fock'][i] = out['fock'][i] + fi

# ====================================================
# Build ONE loss term per molecule with full grad tracking
# ====================================================
Nocc = samples[0]['nocc']
S_tensors = [torch.tensor(s['s1e'], dtype=torch.float64, device=DEVICE) for s in samples]
H_gt_tensors = [torch.tensor(s['H'], dtype=torch.float64, device=DEVICE) for s in samples]
H_pred_tensors = [out['pred_hamiltonian'][i] for i in range(len(samples))]

# ====================================================
# Compute eigh with native vs StableEigh and check grad
# ====================================================
print(f"\n{'='*70}")
print(f"  Per-step grad-norm breakdown in grassmann chain")
print(f"{'='*70}")

for mol_idx in range(1):  # just first mol, representative
    H_pred = H_pred_tensors[mol_idx]
    H_gt = H_gt_tensors[mol_idx]
    S = S_tensors[mol_idx]
    nocc = samples[mol_idx]['nocc']

    # ---- Step A: eigh with torch.linalg.eigh (native) ----
    def compute_grass_native(H_pred):
        s_vals, s_vecs = torch.linalg.eigh(S)
        s_vals = torch.clamp(s_vals, min=EPS_EIGH)
        X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
        
        # gt
        Fs_gt = X.T @ H_gt @ X
        _, eV_gt = torch.linalg.eigh(Fs_gt)
        C_gt_occ = (X @ eV_gt)[:, :nocc]
        
        # pred
        Fs_pred = X.T @ H_pred @ X
        _, eV_pred = torch.linalg.eigh(Fs_pred)
        C_pred_occ = (X @ eV_pred)[:, :nocc]
        
        M = C_gt_occ.T @ S @ C_pred_occ
        _, sigma, _ = torch.linalg.svd(M)
        theta = stable_acos(sigma.clamp(-1, 1))
        return (theta ** 2).sum()

    # ---- Step B: eigh with StableEigh ----
    def compute_grass_stable(H_pred):
        s_vals, s_vecs = torch.linalg.eigh(S)
        s_vals = torch.clamp(s_vals, min=EPS_EIGH)
        X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
        
        # gt (no grad needed)
        with torch.no_grad():
            Fs_gt = X.T @ H_gt @ X
            _, eV_gt = torch.linalg.eigh(Fs_gt)
            C_gt_occ = (X @ eV_gt)[:, :nocc]
        
        # pred
        Fs_pred = X.T @ H_pred @ X
        evals, evecs = StableEigh.apply(Fs_pred, 3.0, nocc)
        C_pred_occ = (X @ evecs)[:, :nocc]
        
        M = C_gt_occ.T @ S @ C_pred_occ
        _, sigma, _ = torch.linalg.svd(M)
        theta = stable_acos(sigma.clamp(-1, 1))
        return (theta ** 2).sum()

    # ---- Test 0: eigh only, no SVD/acos (baseline) ----
    def compute_eigh_only(H_pred, S, H_gt, nocc):
        s_vals, s_vecs = torch.linalg.eigh(S)
        s_vals = torch.clamp(s_vals, min=EPS_EIGH)
        X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
        Fs_pred = X.T @ H_pred @ X
        _, evecs = torch.linalg.eigh(Fs_pred)
        C_pred_occ = (X @ evecs)[:, :nocc]
        return (C_pred_occ ** 2).sum()

    # ---- Test 1: Native eigh → acos → SVD ----
    model.zero_grad()
    loss_native = compute_grass_native(H_pred)
    grads = torch.autograd.grad(loss_native, list(model.parameters()), retain_graph=True)
    eigh_grad_native = sum(g.data.norm(2).item() ** 2 for g in grads if g is not None) ** 0.5

    # ---- Test 2: StableEigh → acos → SVD ----
    model.zero_grad()
    loss_stable = compute_grass_stable(H_pred)
    grads = torch.autograd.grad(loss_stable, list(model.parameters()), retain_graph=True)
    eigh_grad_stable = sum(g.data.norm(2).item() ** 2 for g in grads if g is not None) ** 0.5

    # ---- Test 3: eigh only, no SVD/acos ----
    model.zero_grad()
    loss_eigh_only = compute_eigh_only(H_pred, S, H_gt, nocc)
    grads = torch.autograd.grad(loss_eigh_only, list(model.parameters()), retain_graph=True)
    gn_eigh_only = sum(g.data.norm(2).item() ** 2 for g in grads if g is not None) ** 0.5

    # ---- Compute σ gaps (cause of SVD instability) ----
    with torch.no_grad():
        s_vals, s_vecs = torch.linalg.eigh(S)
        s_vals = torch.clamp(s_vals, min=EPS_EIGH)
        X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
        Fs_pred = X.T @ H_pred @ X
        _, evecs = torch.linalg.eigh(Fs_pred)
        C_pred_occ = (X @ evecs)[:, :nocc]
        Fs_gt = X.T @ H_gt @ X
        _, eV_gt = torch.linalg.eigh(Fs_gt)
        C_gt_occ = (X @ eV_gt)[:, :nocc]
        M = C_gt_occ.T @ S @ C_pred_occ
        _, sigma, _ = torch.linalg.svd(M)
        gaps = torch.diff(torch.sort(sigma)[0])
        min_gap = gaps.min().item() if len(gaps) > 0 else 0

    print(f"\n  [mol {mol_idx}, nocc={nocc}]")
    print(f"  σ range: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
    print(f"  min σ gap: {min_gap:.6f}  →  1/gap² = {1/max(min_gap,1e-8)**2:.1e}")
    print(f"")
    print(f"  grad_norm(eigh only, no SVD/acos):  {gn_eigh_only:.4f}")
    print(f"  grad_norm(full chain, native eigh): {eigh_grad_native:.4f}")
    print(f"  grad_norm(full chain, StableEigh):  {eigh_grad_stable:.4f}")
    print(f"")
    print(f"  SVD+acos extra contribution:")
    print(f"    = grad(full chain) - grad(eigh only)")
    print(f"    = {eigh_grad_native:.4f} - {gn_eigh_only:.4f}")
    print(f"    ≈ {eigh_grad_native - gn_eigh_only:.4f} ({'DOMINANT' if (eigh_grad_native-gn_eigh_only)/max(eigh_grad_native,1e-6) > 0.5 else 'minor'})")
    
    if eigh_grad_stable < eigh_grad_native * 0.8:
        print(f"  ⚠️  StableEigh reduces grad by {1-eigh_grad_stable/eigh_grad_native:.0%}")
    else:
        print(f"  ✅ StableEigh similar to native eigh in full chain")

    # ---- Test 4: Compare projection metric (no SVD/acos) vs geodesic ----
    def compute_grass_proj(H_pred):
        s_vals, s_vecs = torch.linalg.eigh(S)
        s_vals = torch.clamp(s_vals, min=EPS_EIGH)
        X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
        with torch.no_grad():
            Fs_gt = X.T @ H_gt @ X
            _, eV_gt = torch.linalg.eigh(Fs_gt)
            C_gt_occ = (X @ eV_gt)[:, :nocc]
        Fs_pred = X.T @ H_pred @ X
        _, evecs = torch.linalg.eigh(Fs_pred)
        C_pred_occ = (X @ evecs)[:, :nocc]
        P_pred = C_pred_occ @ C_pred_occ.T @ S
        P_gt = C_gt_occ @ C_gt_occ.T @ S
        return ((P_pred - P_gt) ** 2).sum() / nocc

    model.zero_grad()
    loss_proj = compute_grass_proj(H_pred)
    grads = torch.autograd.grad(loss_proj, list(model.parameters()), retain_graph=False)
    gn_proj = sum(g.data.norm(2).item() ** 2 for g in grads if g is not None) ** 0.5
    
    print(f"\n  ======== Metric comparison ========")
    print(f"  geodesic: loss={loss_stable.item():.4f}  grad_norm={eigh_grad_stable:.4f}")
    print(f"  projection:loss={loss_proj.item():.4f}  grad_norm={gn_proj:.4f}")
    print(f"  ratio(geodesic/projection) = {eigh_grad_stable/max(gn_proj,1e-6):.1f}x")

print(f"{'='*70}")
