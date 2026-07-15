"""Diagnose: load earliest ckpt from each geodesic run, compute hami/geodesic/proj loss.

Checks if loss values match the training logs and if there's a code bug.
"""
import sys, os, re, pickle
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
from src.training.losses import stable_acos

# ============================================================
# Replicate the training code's computation exactly
# ============================================================
def solve_eigh(H, S, eps=1e-8):
    s_vals, s_vecs = torch.linalg.eigh(S)
    s_vals = torch.where(s_vals > eps, s_vals, eps)
    X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
    Hp = X.T @ H @ X
    e_vals, e_vecs = torch.linalg.eigh(Hp)
    C = X @ e_vecs
    return e_vals, C

def grassmann_projection(C_gt, C_pred, S):
    M = C_gt.T @ S @ C_pred
    return C_gt.shape[1] - torch.linalg.matrix_norm(M, ord='fro') ** 2

def grassmann_densityS(C_gt, C_pred, S):
    P_pred = C_pred @ C_pred.T @ S
    P_gt = C_gt @ C_gt.T @ S
    return ((P_pred - P_gt) ** 2).sum() / C_gt.shape[1]

def grassmann_geodesic(C_gt, C_pred, S, use_stable_acos=True):
    """Exact same formula as training code: sum(arccos(sigma)^2)."""
    M = C_gt.T @ S @ C_pred
    _, s, _ = torch.linalg.svd(M)
    if use_stable_acos:
        theta = stable_acos(s.clamp(-1, 1))
    else:
        theta = torch.acos(s.clamp(-1, 1))
    return (theta ** 2).sum()

def stationarity_loss(H, C, S):
    D = 2 * C @ C.T
    delta = H @ D @ S - S @ D @ H
    return (delta ** 2).sum()

# ============================================================
# Load model from checkpoint
# ============================================================
def load_model(ckpt_path, DEVICE):
    ckpt = torch.load(ckpt_path, map_location='cpu')
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
    for req_key in ['ckpt_path', 'cutoff_upper', 'output_model', 'enable_energy', 'enable_forces', 'enable_symmetry']:
        if req_key not in hp_dict or hp_dict[req_key] is None:
            hp_dict[req_key] = '/tmp/ckpts' if req_key == 'ckpt_path' else (
                None if req_key == 'output_model' else False)
    hp_dict['remove_init'] = True
    model = create_model(hp_dict)
    fixed_sd = {re.sub(r'^model\.', '', k): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(fixed_sd, strict=True)
    model.to(DEVICE).eval()
    return model, hp_dict

# ============================================================
# Load a few ethanol samples from lmdb
# ============================================================
def load_samples(dataset_path, split_path, n_mol=8):
    import lmdb
    env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=32)
    split = torch.load(split_path, map_location='cpu', weights_only=False)
    if isinstance(split, (list, tuple)):
        test_ids = split[2].tolist()[:n_mol]
    else:
        test_ids = split.tolist()[:n_mol]
    samples = []
    with env.begin() as txn:
        for abs_idx in test_ids:
            raw = pickle.loads(txn.get(int(abs_idx).to_bytes(4, 'big')))

            def _to_numpy(x):
                if isinstance(x, bytes):
                    # raw buffer: atoms=int32, pos=float64, Ham=float64
                    return np.frombuffer(x, dtype=np.float64).copy()
                if hasattr(x, 'numpy'): return x.numpy()
                return x

            def _atoms_to_numpy(x):
                if isinstance(x, bytes):
                    return np.frombuffer(x, dtype=np.int32).copy()
                if hasattr(x, 'numpy'): return x.numpy()
                return x

            atoms = _atoms_to_numpy(raw['atoms'])
            pos = _to_numpy(raw['pos'])
            H = _to_numpy(raw.get('Ham', raw.get('hamiltonian')))
            H_init = _to_numpy(raw['Ham_init'])
            labels = _to_numpy(raw['labels'])
            edge_index = _to_numpy(raw['edge_index'])

            # Reshape pos and H from flat buffers
            n_atoms = len(atoms)
            pos = pos.reshape(n_atoms, 3) if pos.ndim == 1 else pos
            norb = H_init.shape[0]
            H = H.reshape(norb, norb) if H.ndim == 1 else H

            nelec = int(np.sum(atoms))
            nocc = nelec // 2
            samples.append({
                'idx': abs_idx, 'cid': raw['id'],
                'atoms': atoms, 'pos': pos,
                'H': H, 'H_init': H_init,
                'norb': norb, 'nocc': nocc,
                'labels': labels, 'edge_index': edge_index,
                'num_nodes': raw['num_nodes'],
            })
    env.close()
    return samples

# ============================================================
# Main diagnostic
# ============================================================
def main():
    DEVICE = 'cpu'
    DATASET = "/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb"
    SPLIT = "/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/MD17_ethanol_trainset_with_5000_data.pt"

    RUNS = {
        'w0.0 (control)': {
            'ckpt': '/home/pepe/workbench/grassd_outputs/v2_geodesic_w0.0_20260715_112558/logs/first10-epoch00-0.000858.ckpt',
            'log_hami_mae': 0.00086,
            'log_grassmann': 1.14,
        },
        'w0.05': {
            'ckpt': '/home/pepe/workbench/grassd_outputs/v2_geodesic_w0.05_20260715_112607/logs/first10-epoch00-1.232646.ckpt',
            'log_hami_mae': 0.0075,
            'log_grassmann': 24.50,
        },
        'w1.0': {
            'ckpt': '/home/pepe/workbench/grassd_outputs/v2_geodesic_w1.0_20260715_112601/logs/first10-epoch00-31.834799.ckpt',
            'log_hami_mae': 0.099,
            'log_grassmann': 31.68,
        },
    }

    print("=" * 75)
    print("  Diagnostic: ckpt loss values vs training logs")
    print("=" * 75)

    collate = collate_fn_unified(long_cutoff_upper=9, unit=1)
    conv, _, mask, _ = get_conv_variable_lin('def2-svp')
    max_block_size = conv.max_block_size
    print(f"  Loading {len(RUNS)} models + data samples ...")

    # Load samples once
    samples = load_samples(DATASET, SPLIT, n_mol=10)
    print(f"  Loaded {len(samples)} molecules from ethanol def2-svp")

    for run_name, run_info in RUNS.items():
        print(f"\n{'─' * 75}")
        print(f"  [{run_name}]")
        ckpt_path = run_info['ckpt']
        print(f"  ckpt: {os.path.basename(ckpt_path)}")

        model, hp = load_model(ckpt_path, DEVICE)
        print(f"  grassmann_weight={hp.get('grassmann_weight','?')}  metric={hp.get('grassmann_metric','?')}")

        hami_list = []
        geo_list_stable = []
        geo_list_native = []
        proj_list = []
        dens_list = []

        for sample in samples:
            from pyscf import gto
            mol = gto.M(atom=[(int(z), sample['pos'][i]) for i, z in enumerate(sample['atoms'])],
                        basis='def2-svp', unit='ang', verbose=0)
            s1e = mol.intor('int1e_ovlp')
            norb = sample['norb']
            nocc = sample['nocc']

            entry = {
                'idx': 0, 'cid': sample['cid'],
                'pos': sample['pos'], 'atomic_numbers': sample['atoms'],
                'molecule_size': len(sample['pos']),
                'fock': sample['H'] - sample['H_init'],
                'fock_init': sample['H_init'].copy(),
                'C_init': np.eye(norb),
                'D_gt': np.eye(norb),
                'overlap': s1e,
                'n_orb': 0,
                'labels': sample['labels'],
                'edge_index': sample['edge_index'],
                'buildblock_mask': mask, 'max_block_size': max_block_size,
                'grid_coords': np.array([0], dtype=np.float64),
                'grid_weights': np.array([0], dtype=np.float64),
                'num_nodes': sample['num_nodes'],
            }

            batch = collate([entry]).to(DEVICE)

            with torch.no_grad():
                out = model(batch)
                if 'pred_hamiltonian_diagonal_blocks' not in out:
                    out = model.hami_model(out)
                out = model.hami_model.build_final_matrix(out, sym_type='sym')

                fock_init_t = torch.tensor(sample['H_init'], dtype=torch.float64, device=DEVICE)
                H_pred = out['pred_hamiltonian'][0] + fock_init_t
                H_gt = out['fock'][0] + fock_init_t

                # Hami MAE
                delta = H_pred - H_gt
                hami_list.append(float(delta.abs().mean().cpu()))

                # Overlap
                S = torch.tensor(s1e, dtype=torch.float64, device=DEVICE)

                # Solve eigh
                e_gt, C_gt = solve_eigh(H_gt, S)
                e_pred, C_pred = solve_eigh(H_pred, S)
                C_gt_occ = C_gt[:, :nocc]
                C_pred_occ = C_pred[:, :nocc]

                # Grassmann metrics
                geo_list_stable.append(float(grassmann_geodesic(C_gt_occ, C_pred_occ, S, use_stable_acos=True)))
                geo_list_native.append(float(grassmann_geodesic(C_gt_occ, C_pred_occ, S, use_stable_acos=False)))
                proj_list.append(float(grassmann_projection(C_gt_occ, C_pred_occ, S)))
                dens_list.append(float(grassmann_densityS(C_gt_occ, C_pred_occ, S)))

        # Print per-sample
        print(f"\n  {'mol':>4s}  {'nocc':>4s}  {'hami_mae':>12s}  {'geodesic(s)':>14s}  {'geodesic(n)':>14s}  {'projection':>14s}  {'densityS':>14s}")
        for i in range(len(samples)):
            print(f"  {i:4d}  {samples[i]['nocc']:4d}  "
                  f"{hami_list[i]:12.6e}  {geo_list_stable[i]:14.6f}  "
                  f"{geo_list_native[i]:14.6f}  {proj_list[i]:14.6f}  {dens_list[i]:14.6f}")

        # Summary
        mean_hami = np.mean(hami_list)
        mean_geo_s = np.mean(geo_list_stable)
        mean_geo_n = np.mean(geo_list_native)
        mean_proj = np.mean(proj_list)
        mean_dens = np.mean(dens_list)

        print(f"\n  {'Mean':>4s}       {mean_hami:12.6e}  {mean_geo_s:14.6f}  "
              f"{mean_geo_n:14.6f}  {mean_proj:14.6f}  {mean_dens:14.6f}")

        # Compare with logs
        print(f"\n  [Comparison with training log]")
        print(f"    hami_mae:        local={mean_hami:.6e}  log≈{run_info['log_hami_mae']:.6e}")
        print(f"    grassmann_loss:  local={mean_geo_s:.4f}  log≈{run_info['log_grassmann']:.4f}")

        # Check stable_acos vs native acos (forward should be identical)
        acos_diff = max(abs(g_s - g_n) for g_s, g_n in zip(geo_list_stable, geo_list_native))
        print(f"    stable_acos vs torch.acos max diff = {acos_diff:.2e} (forward should be ~0)")

        # Check if the loss range makes sense
        max_possible = np.mean([s['nocc'] * (np.pi / 2) ** 2 for s in samples])
        print(f"    max possible geodesic² = {max_possible:.4f},  current = {mean_geo_s:.4f}  ({mean_geo_s/max_possible*100:.1f}%)")

    print(f"\n{'=' * 75}")
    print("  Done.")

if __name__ == '__main__':
    main()
