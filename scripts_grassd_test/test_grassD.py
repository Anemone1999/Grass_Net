import sys, os, re, json, argparse, pickle, lmdb, warnings, time
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

SRC = "/home/pepe/codebench/GrassD_sphnet_ver2/src"
sys.path.insert(0, os.path.join(SRC, os.pardir))
sys.path.insert(0, SRC)
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf
from src.dataset.buildblock import get_conv_variable_lin
from src.dataset.utils import collate_fn_unified
from src.models.model import create_model
from src.utility.pyscf import get_homo_lumo_from_h, get_pyscf_obj_from_dataset

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def solve_eigh(H, S, eps=1e-8):
    """Solve generalized eigenvalue problem H C = S C E"""
    s_vals, s_vecs = torch.linalg.eigh(S)
    s_vals = torch.where(s_vals > eps, s_vals, eps)
    X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
    Hp = X.T @ H @ X
    e_vals, e_vecs = torch.linalg.eigh(Hp)
    C = X @ e_vecs
    return e_vals, C

def grassmann_projection(C_gt, C_pred, S):
    """nocc - ||C_gt^T @ S @ C_pred||_F^2"""
    M = C_gt.T @ S @ C_pred
    return C_gt.shape[1] - torch.linalg.matrix_norm(M, ord='fro') ** 2

def grassmann_densityS(C_gt, C_pred, S):
    """||C_pred @ C_pred^T @ S - C_gt @ C_gt^T @ S||_F^2 / nocc"""
    P_pred = C_pred @ C_pred.T @ S
    P_gt = C_gt @ C_gt.T @ S
    return ((P_pred - P_gt) ** 2).sum() / C_gt.shape[1]

def grassmann_geodesic(C_gt, C_pred, S):
    """sum(theta^2) from SVD of C_gt^T @ S @ C_pred"""
    M = C_gt.T @ S @ C_pred
    _, s, _ = torch.linalg.svd(M)
    theta = torch.acos(s.clamp(-1, 1))
    return (theta ** 2).sum()

def stationarity_loss(H, C, S):
    """||F @ D @ S - S @ D @ F||_F^2 where D = 2*C_occ@C_occ^T"""
    D = 2 * C @ C.T
    delta = H @ D @ S - S @ D @ H
    return (delta ** 2).sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default="/home/pepe/workbench/yfversion/checkpoints/baseline_nosparse/logs/step=199999-epoch=61-val_loss=4.096361600087507e-05.ckpt")
    parser.add_argument('--dataset', default="/home/pepe/data/QH9Stable/full_data")
    parser.add_argument('--split', default="/home/pepe/data/QH9Stable/full_data/processed_QH9Stable_size_ood.pt")
    parser.add_argument('--basis', default='def2-svp')
    parser.add_argument('--n_mol', type=int, default=100)
    parser.add_argument('--out', default="/home/pepe/workbench/grassd_outputs")
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = os.path.join(args.out, f"test_grassD_{timestamp}{tag}")
    os.makedirs(out_dir, exist_ok=True)

    # ===== Load model =====
    print(f"[1/4] Loading model from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location='cpu')
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
            hp_dict[req_key] = '/tmp/ckpts' if req_key == 'ckpt_path' else (None if req_key in ['output_model'] else False)
    hp_dict['remove_init'] = True

    model = create_model(hp_dict)
    fixed_sd = {re.sub(r'^model\.', '', k): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(fixed_sd, strict=True)
    model.to(DEVICE).eval()

    conv, _, mask, _ = get_conv_variable_lin(args.basis)
    max_block_size = conv.max_block_size
    collate = collate_fn_unified(long_cutoff_upper=9, unit=1)

    # ===== Load split =====
    print(f"[2/4] Loading split from {args.split} ...")
    split = torch.load(args.split, map_location='cpu', weights_only=False)
    test_ids = split[2].tolist()
    if args.n_mol > 0:
        test_ids = test_ids[:args.n_mol]
    print(f"  Test set total: {len(split[2])}, using {len(test_ids)} samples")

    # ===== Inference =====
    print(f"[3/4] Running inference on {len(test_ids)} molecules ...")
    env = lmdb.open(args.dataset, readonly=True, lock=False, readahead=False, meminit=False, max_readers=32)

    results = []
    errors = []

    for lin_idx, abs_idx in enumerate(test_ids):
        with env.begin() as txn:
            raw = pickle.loads(txn.get(int(abs_idx).to_bytes(4, 'big')))

        atoms = raw['atoms'].numpy() if hasattr(raw['atoms'], 'numpy') else raw['atoms']
        pos = raw['pos'].numpy() if hasattr(raw['pos'], 'numpy') else raw['pos']
        Ham = raw['Ham'].numpy() if hasattr(raw['Ham'], 'numpy') else raw['Ham']
        norbs = Ham.shape[0]
        H_init = raw['Ham_init'].numpy() if hasattr(raw['Ham_init'], 'numpy') else raw['Ham_init']
        labels = raw['labels'].numpy() if hasattr(raw['labels'], 'numpy') else raw['labels']
        edge_index = raw['edge_index'].numpy() if hasattr(raw['edge_index'], 'numpy') else raw['edge_index']

        from pyscf import gto
        mol = gto.M(atom=[(int(z), pos[i]) for i, z in enumerate(atoms)],
                    basis=args.basis, unit='ang', verbose=0)
        s1e = mol.intor('int1e_ovlp')

        nelec = int(np.sum(atoms))
        nocc = nelec // 2

        entry = {
            'idx': lin_idx, 'cid': raw['id'],
            'pos': pos, 'atomic_numbers': atoms,
            'molecule_size': len(pos),
            'fock': Ham - H_init,
            'fock_init': H_init.copy(),
            'C_init': np.eye(norbs),
            'D_gt': raw.get('D_gt', np.eye(norbs)).numpy() if hasattr(raw.get('D_gt', None), 'numpy') else raw.get('D_gt', np.eye(norbs)),
            'overlap': s1e,
            'n_orb': 0,
            'labels': labels,
            'edge_index': edge_index,
            'buildblock_mask': mask, 'max_block_size': max_block_size,
            'grid_coords': np.array([0], dtype=np.float64),
            'grid_weights': np.array([0], dtype=np.float64),
            'num_nodes': raw['num_nodes'],
        }

        try:
            batch = collate([entry]).to(DEVICE)
        except Exception as e:
            errors.append({"idx": abs_idx, "cid": int(raw['id']), "error": str(e), "stage": "collate"})
            continue

        with torch.no_grad():
            try:
                out = model(batch)
                if 'pred_hamiltonian_diagonal_blocks' not in out:
                    out = model.hami_model(out)
                out = model.hami_model.build_final_matrix(out, sym_type='sym')

                # Add back fock_init
                fock_init_t = torch.tensor(H_init, dtype=torch.float64, device=DEVICE)
                H_pred = out['pred_hamiltonian'][0] + fock_init_t
                H_gt = out['fock'][0] + fock_init_t

                # ---- Hami MAE ----
                delta = H_pred - H_gt
                hami_mae = float(delta.abs().mean().cpu())
                hami_maxae = float(delta.abs().max().cpu())
                hami_rmse = float((delta ** 2).mean().sqrt().cpu())

                # ---- Overlap matrix to device ----
                S = torch.tensor(s1e, dtype=torch.float64, device=DEVICE)

                # ---- Solve generalized eigenvalue problem ----
                e_gt, C_gt = solve_eigh(H_gt, S)
                e_pred, C_pred = solve_eigh(H_pred, S)

                C_gt_occ = C_gt[:, :nocc]
                C_pred_occ = C_pred[:, :nocc]

                # ---- Grassmann metrics ----
                g_proj = float(grassmann_projection(C_gt_occ, C_pred_occ, S).cpu())
                g_dens = float(grassmann_densityS(C_gt_occ, C_pred_occ, S).cpu())
                g_geod = float(grassmann_geodesic(C_gt_occ, C_pred_occ, S).cpu())

                # ---- Stationarity ----
                stat = float(stationarity_loss(H_gt, C_pred_occ, S).cpu())

                # ---- MO coefficient cosine similarity ----
                cos_sim = torch.cosine_similarity(C_pred_occ, C_gt_occ, dim=0).abs().mean().item()

                # ---- MO energy MAE ----
                mo_e_mae = float((e_pred[:nocc] - e_gt[:nocc]).abs().mean().cpu())

                results.append({
                    'cid': int(raw['id']), 'n_atoms': int(raw['num_nodes']),
                    'n_orbitals': norbs, 'n_occ': nocc,
                    'hami_mae': hami_mae, 'hami_maxae': hami_maxae,
                    'hami_rmse': hami_rmse,
                    'grass_projection': g_proj,
                    'grass_densityS': g_dens,
                    'grass_geodesic': g_geod,
                    'stationarity': stat,
                    'c_cos_sim': cos_sim,
                    'mo_energy_mae': mo_e_mae,
                })

            except Exception as e:
                import traceback
                errors.append({"idx": abs_idx, "cid": int(raw['id']), "error": str(e), "stage": "inference"})
                traceback.print_exc()

        if (lin_idx + 1) % 20 == 0 or lin_idx == len(test_ids) - 1:
            n_ok = len(results)
            if n_ok > 0:
                r = results[-1]
                print(f'  {lin_idx+1}/{len(test_ids)}  ok={n_ok}  '
                      f'H_MAE={r["hami_mae"]:.3e}  C_cos={r["c_cos_sim"]:.6f}  '
                      f'Grass_proj={r["grass_projection"]:.3e}')

    env.close()

    # ===== Save =====
    print(f"[4/4] Saving results to {out_dir} ...")
    n_ok = len(results)

    summary = {
        'ckpt': args.ckpt, 'dataset': args.dataset,
        'basis': args.basis, 'n_total': len(test_ids),
        'n_ok': n_ok, 'n_fail': len(test_ids) - n_ok,
    }
    # only mean for non-GrassD metrics
    for key in ['hami_mae', 'hami_maxae', 'hami_rmse',
                'stationarity', 'c_cos_sim', 'mo_energy_mae']:
        vals = [r[key] for r in results]
        summary[f'mean_{key}'] = float(np.mean(vals))
    # mean + std + median for Grassmann metrics
    for key in ['grass_projection', 'grass_densityS', 'grass_geodesic']:
        vals = [r[key] for r in results]
        summary[f'mean_{key}'] = float(np.mean(vals))
        summary[f'median_{key}'] = float(np.median(vals))
        summary[f'std_{key}'] = float(np.std(vals))

    out_path = os.path.join(out_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(dict(summary=summary, results=results, errors=errors),
                  f, indent=2, cls=NumpyEncoder)

    print(f'\n=== Results: {out_path} ===')
    print(f'  Mean H MAE:              {summary["mean_hami_mae"]:.6e}')
    print(f'  Mean C cos sim:          {summary["mean_c_cos_sim"]:.6f}')
    print(f'  Mean MO energy MAE:      {summary["mean_mo_energy_mae"]:.6e}')
    print(f'  Mean Stationarity:       {summary["mean_stationarity"]:.6e}')
    print(f'  Grassmann(proj):         mean={summary["mean_grass_projection"]:.6e}  median={summary["median_grass_projection"]:.6e}  std={summary["std_grass_projection"]:.6e}')
    print(f'  Grassmann(densityS):     mean={summary["mean_grass_densityS"]:.6e}  median={summary["median_grass_densityS"]:.6e}  std={summary["std_grass_densityS"]:.6e}')
    print(f'  Grassmann(geodesic):     mean={summary["mean_grass_geodesic"]:.6e}  median={summary["median_grass_geodesic"]:.6e}  std={summary["std_grass_geodesic"]:.6e}')
    print('Done!')

if __name__ == '__main__':
    main()
