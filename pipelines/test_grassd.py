'''
Test Grassmann distances on ethanol/qzvp models.
Usage:
  python pipelines/test_grassd.py --ckpt <path> --dataset <lmdb> --index <pt> \
      --basis def2-qzvp --n_test 25 --out <dir>
'''
import sys, os, re, json, argparse, pickle, lmdb, warnings
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf
from src.dataset.buildblock import get_conv_variable_lin
from src.dataset.utils import collate_fn_unified
from src.models.model import create_model
from src.training.losses import GrassmannError

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

Z_TO_SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--basis', default='def2-qzvp')
    parser.add_argument('--n_test', type=int, default=25)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model_name', default='unknown')
    parser.add_argument('--hami_only', action='store_true', help='skip grassmann, only compute H MAE')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    # --- load model ---
    print(f'Loading {args.ckpt} ...')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    hp = ckpt['hyper_parameters']
    try:
        hp_dict = OmegaConf.to_container(hp, resolve=True, throw_on_missing=False)
    except Exception:
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
                None if req_key in ['output_model'] else False)
    hp_dict['remove_init'] = True

    model = create_model(hp_dict)
    fixed_sd = {re.sub(r'^model\.', '', k): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(fixed_sd, strict=True)
    model.to(DEVICE).eval()

    # --- blocks ---
    conv, _, mask, _ = get_conv_variable_lin(args.basis)
    max_block_size = conv.max_block_size
    collate = collate_fn_unified(long_cutoff_upper=9, unit=1)

    # --- test indices ---
    idx = torch.load(args.index, map_location='cpu', weights_only=False)
    if args.n_test < 0:
        test_ids = idx[2].tolist()
    else:
        test_ids = idx[2].tolist()[:args.n_test]
    print(f'Test: {len(test_ids)} samples')

    if args.hami_only:
        metrics = []
        grass_calcs = {}
    else:
        metrics = ['projection', 'densityS', 'geodesic']
        grass_calcs = {
            m: GrassmannError(enable_grassmann=True, enable_stationarity=False,
                              grassmann_weight=1.0, stationarity_weight=0.0,
                              basis=args.basis, ed_type='naive', grassmann_metric=m)
            for m in metrics
        }

    # --- run ---
    env = lmdb.open(args.dataset, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False, max_readers=32)

    all_grass = {m: [] for m in metrics} if not args.hami_only else {}
    hami_maes = []

    for lin_idx, abs_idx in enumerate(test_ids):
        with env.begin() as txn:
            raw = pickle.loads(txn.get(int(abs_idx).to_bytes(4, 'big')))

        try: atoms = np.frombuffer(raw['atoms'], np.int32)
        except: atoms = raw['atoms']
        try: pos = np.frombuffer(raw['pos'], np.float64).reshape(raw['num_nodes'], 3)
        except: pos = raw['pos']
        try:
            Ham = np.frombuffer(raw['Ham'], np.float64)
            norbs = int(np.sqrt(Ham.shape[0]))
            Ham = Ham.reshape(norbs, norbs)
        except:
            Ham = raw['Ham']; norbs = Ham.shape[0]
        H_init = raw['Ham_init']

        if not args.hami_only:
            from pyscf import gto
            mol = gto.M(atom=[[Z_TO_SYM[int(z)], pos[i]] for i, z in enumerate(atoms)],
                        basis=args.basis, unit='ang', verbose=0)
            s1e = mol.intor('int1e_ovlp')
        else:
            s1e = np.eye(norbs)

        entry = {
            'idx': lin_idx, 'cid': raw['id'],
            'pos': pos, 'atomic_numbers': atoms,
            'molecule_size': len(pos),
            'fock': Ham - H_init,
            'fock_init': H_init.copy(),
            'C_init': np.eye(norbs),
            'D_gt': raw.get('D_gt', np.eye(norbs)),
            'overlap': s1e,
            'n_orb': 0,
            'labels': raw['labels'].numpy() if hasattr(raw['labels'], 'numpy') else raw['labels'],
            'edge_index': raw['edge_index'].numpy() if hasattr(raw['edge_index'], 'numpy') else raw['edge_index'],
            'buildblock_mask': mask, 'max_block_size': max_block_size,
            'grid_coords': 0, 'grid_weights': 0, 'num_nodes': raw['num_nodes'],
        }

        batch = collate([entry]).to(DEVICE)
        with torch.no_grad():
            out = model(batch)
            out = model.hami_model.build_final_matrix(out, sym_type='sym')
            for i in range(len(out['pred_hamiltonian'])):
                out['fock'][i] = out['fock'][i] + out['fock_init'][i]
                out['pred_hamiltonian'][i] = out['pred_hamiltonian'][i] + out['fock_init'][i]
            if not args.hami_only:
                out['s1e'] = [s1e]

            # Hami MAE
            hp_np = out['pred_hamiltonian'][0].cpu().numpy()
            hg_np = out['fock'][0].cpu().numpy()
            hami_maes.append(float(np.abs(hp_np - hg_np).mean()))

            # Grassmann (skip if hami_only)
            if not args.hami_only:
                for m in metrics:
                    ed = {'loss': 0}
                    grass_calcs[m].cal_loss(out, ed)
                    gv = ed.get('grassmann_loss', 0)
                    if isinstance(gv, torch.Tensor): gv = gv.item()
                    all_grass[m].append(gv)

        if (lin_idx + 1) % 200 == 0 or (args.hami_only and (lin_idx + 1) % 500 == 0):
            print(f'  {lin_idx+1}/{len(test_ids)}')

    env.close()

    # --- save ---
    summary = {
        'model': args.model_name, 'ckpt': args.ckpt,
        'basis': args.basis, 'n_test': args.n_test,
        'mean_hami_mae': float(np.mean(hami_maes)),
    }
    for m in metrics:
        vals = all_grass[m]
        summary[f'grass_{m}_mean'] = float(np.mean(vals))
        summary[f'grass_{m}_median'] = float(np.median(vals))
        summary[f'grass_{m}_std'] = float(np.std(vals))

    with open(os.path.join(args.out, f'grassmann_{args.model_name}.json'), 'w') as f:
        json.dump(dict(summary=summary, per_frame={
            'hami_mae': hami_maes,
            **{f'grass_{m}': all_grass[m] for m in metrics}
        }), f, indent=2, cls=NumpyEncoder)

    print(f'\n=== {args.model_name} ===')
    print(f'  mean H MAE: {summary["mean_hami_mae"]:.6e}')
    for m in metrics:
        print(f'  grass_{m:12s}: mean={summary[f"grass_{m}_mean"]:.6e}  median={summary[f"grass_{m}_median"]:.6e}')
    print(f'\nOutput: {os.path.join(args.out, f"grassmann_{args.model_name}.json")}')

if __name__ == '__main__':
    main()
