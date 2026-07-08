
import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
import pdb
from torch.autograd import Function
from ..utility.eigen_solver import ED_trunc_p

def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_ratio = (
        isinstance(train_size, float) and train_size <= 1,
        isinstance(val_size, float)  and val_size <= 1,
        isinstance(test_size, float) and test_size <= 1,
    )
    if train_size:
        train_size = round(dset_len * train_size) if is_ratio[0] else round(train_size)
    if val_size:
        val_size = round(dset_len * val_size) if is_ratio[1] else round(val_size)
    if test_size:
        test_size = round(dset_len * test_size) if is_ratio[2] else round(test_size)

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_ratio[2]:
            test_size -= 1
        elif is_ratio[1]:
            val_size -= 1
        elif is_ratio[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int32)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(
    dataset_len,
    train_num,
    val_num,
    test_num,
    seed,
    filename=None,
    splits=None,
    order=None,
):
    ###
    ##train_num,val_num,test_num coule be int or float percentage.
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_num, val_num, test_num, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )

def check_fock(batch_data, xc, pos_unit, basis):
    
    import cupy as cp
    from gpu4pyscf.dft import rks 
    import pyscf
    mol_size = batch_data["molecule_size"][0].detach().cpu().numpy()
    atoms = batch_data["atomic_numbers"][0:0 + mol_size].detach().cpu().numpy()
    pos = batch_data["pos"][0:0 + mol_size].detach().cpu().numpy()
    single_mol = pyscf.gto.Mole()
    single_mol.atom = [[atoms[i], pos[i]] for i in range(len(atoms))]
    single_mol.unit = pos_unit
    single_mol.basis = basis
    single_mol.build()


    mf = rks.RKS(single_mol).set(xc=xc)
    mf.kernel()

    dm_scf = mf.make_rdm1()
    dm_gt = cp.asarray(batch_data['D_gt'][0]).astype(cp.float64)

    fock_scf = torch.tensor(mf.get_fock(), dtype=torch.float64).to(batch_data['fock'][0].device)

    energy_scf = mf.energy_tot(dm=dm_scf)
    energy_gt = mf.energy_tot(dm=dm_gt)

    print(cp.allclose(dm_scf, dm_gt, atol=1e-3), flush=True)
    print(torch.allclose(fock_scf, batch_data['fock'][0].type(fock_scf.dtype), atol=1e-3), flush=True)
    print(cp.mean(cp.abs(dm_scf - dm_gt)))
    print(torch.mean(torch.abs(fock_scf - batch_data['fock'][0].type(fock_scf.dtype))))
    print(cp.abs(energy_scf - energy_gt), flush=True)
    print("===================================", flush=True)

    return

def loss_analyse(batch_data):
    import cupy as cp
    from gpu4pyscf.dft import rks 
    import pyscf
    mol_size = batch_data["molecule_size"][0].detach().cpu().numpy()
    atoms = batch_data["atomic_numbers"][0:0 + mol_size].detach().cpu().numpy()
    pos = batch_data["pos"][0:0 + mol_size].detach().cpu().numpy()
    single_mol = pyscf.gto.Mole()
    single_mol.atom = [[atoms[i], pos[i]] for i in range(len(atoms))]
    single_mol.basis = 'def2-svp'
    single_mol.build()
    mf = rks.RKS(single_mol).set(xc='b3lyp5')

    dm_pred = cp.asarray(batch_data["D_pred"][0])
    dm_gt = cp.asarray(batch_data["D_gt"][0])
    fock = cp.asarray(batch_data["fock"][0])
    overlap = cp.asarray(batch_data["overlap"][0])

    energy_gt = mf.energy_tot(dm=dm_gt)
    energy_pred = mf.energy_tot(dm=dm_pred)
    EA_loss = fock @ dm_pred @ overlap - overlap @ dm_pred @ fock
    trHD_loss = cp.sum(fock * dm_gt) - cp.sum(fock * dm_pred)

    error_mat_dm = (dm_gt - dm_pred).get()
    error_trHD = (fock * (dm_gt - dm_pred)).get()
    plot_mat({
              "DMerror": 
              {"mat": error_mat_dm,
              "label": "DM",
              "max_value_coeff": 0.02},
              "trHDerror":
              {"mat": error_trHD,
               "label": "trHD",
               "max_value_coeff": 0.005}
              }, 
             f"mol_{batch_data.idx.item()}_Xpred", batch_data.idx)

    print("mol id: ", batch_data.idx.item())
    print("atoms: ", batch_data.atomic_numbers)
    print("energy error: ", cp.abs(energy_gt - energy_pred))
    print("DM MAE: ", cp.mean(cp.abs(dm_pred - dm_gt)))
    print("EA loss: ", cp.mean(cp.abs(EA_loss)))
    print("trHD loss: ", cp.abs(trHD_loss))

    pdb.set_trace()
    return

def plot_mat(mat_dict, title, idx, subfix='svg'):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(mat_dict.keys()), figsize=(len(mat_dict.keys()) * 12, 12))
    if len(axes) == 1:
        axes = [axes]

    for i, (key, value) in enumerate(mat_dict.items()):
        range = np.max(value["mat"]) - np.min(value["mat"])
        mean = np.mean(value["mat"])
        var = np.var(value["mat"])
        stat_text = (
            f'Range: {range:.7f}\n'
            f'Mean: {mean:.7f}\n'
            f'Var: {var:.7f}'
        )
        if "max_value_coeff" in value.keys():
            max_value = value["max_value_coeff"]
        else:
            max_value = np.max(np.abs(value["mat"]))

        im = axes[i].imshow(value["mat"], cmap='RdBu_r', vmax=max_value, vmin=-max_value)
        axes[i].set_title(value["label"], fontsize=30)
        plt.colorbar(im, ax=axes[i])
        plt.text(0.5, -0.1, stat_text, fontsize=24,
                 ha='center', va='top', transform=axes[i].transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.1))
        np.savez(f"../outputs_test_loss_analyse/Xpred/{key}_{idx}.npz", value["mat"])
    plt.savefig(f"../outputs_test_loss_analyse/Xpred/{title}.{subfix}")
    return

def eigen_solver(full_hamiltonian, overlap_matrix, ed_type="naive"):
    eig_success = True
    degenerate_eigenvals = False
    if ed_type == "naive": 
        ed_solver = torch.linalg.eigh
    elif ed_type == "trunc":
        ed_solver = ED_trunc_p.apply
    else:
        raise NotImplementedError()
    try:
        eigvals, eigvecs = ed_solver(overlap_matrix)
        eps = 1e-8 * torch.ones_like(eigvals)
        eigvals = torch.where(eigvals > 1e-8, eigvals, eps)
        frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)
        
        Fs = torch.einsum('ij,jk,kl->il', frac_overlap.transpose(-1, -2), full_hamiltonian, frac_overlap)
        Fs = Fs.unsqueeze(0)

        if ed_type == "naive":
            orbital_energies, orbital_coefficients = ed_solver(Fs)
        else:
            orbital_energies, orbital_coefficients = ed_solver(Fs, 0.95)

        _, counts = torch.unique_consecutive(orbital_energies, return_counts=True)
        if torch.any(counts>1): #will give NaNs in backward pass
            degenerate_eigenvals = True #will give NaNs in backward pass

        orbital_energies = orbital_energies.squeeze(0)
        orbital_coefficients = orbital_coefficients.squeeze(0)
        orbital_coefficients = torch.einsum('ij,jk->ik', frac_overlap, orbital_coefficients)
    except RuntimeError:
        eig_success = False
        orbital_energies = None
        orbital_coefficients = None

    return orbital_energies, orbital_coefficients, eig_success, degenerate_eigenvals

def diagonalize_fock(F, S, tol=1e-9):
    # TODO: use zh's eigh instead of torch's eigh
    s_vals, s_vecs = torch.linalg.eigh(S)
    
    mask = s_vals > tol
    s_inv_sqrt = torch.zeros_like(s_vals)
    s_inv_sqrt[mask] = 1.0 / torch.sqrt(s_vals[mask])
    
    X = s_vecs @ torch.diag(s_inv_sqrt) @ s_vecs.t()
    
    F_prime = torch.einsum('ij,jk,kl->il', X.T, F, X)
    epsilon, C_prime = torch.linalg.eigh(F_prime)
    C = X @ C_prime

    return epsilon, C

def cal_D_from_C(C, n_orb):
    occ = torch.zeros_like(C[0])
    occ[:n_orb] = 2
    occ.type(C.dtype)
    I_occ = torch.diag(occ)
    D = torch.einsum('ik,kl,jl->ij', C, I_occ, C)
    return D

def cal_D_from_H(batch_data, mol_idx, flag="gt", ed_type='naive'):
    n_orb = batch_data["n_orb"][mol_idx]
    if flag == "gt":
        Fock = batch_data['fock'][mol_idx]
    else:
        Fock = batch_data['pred_hamiltonian'][mol_idx]
    S = batch_data['overlap'][mol_idx]
    epsilon, C, eig_success, degenerate_eigenvals = eigen_solver(Fock, S, ed_type=ed_type)
    if eig_success and not degenerate_eigenvals:
        D = cal_D_from_C(C, n_orb)
        return D, epsilon
    else:
        print(f"eigen solver failed, eigsuccess: {eig_success}, degenerate_eigenvals: {degenerate_eigenvals}")
        return None, None

def cal_no_redundant_D(batch_data, mol_idx):

    n_orb = batch_data["n_orb"][mol_idx]

    C = batch_data['C_init'][mol_idx]
    S = batch_data['overlap'][mol_idx]
    C_inv = torch.einsum('ij, jk -> ik', C.T, S)
    C_T_inv = torch.einsum('ij, jk -> ik', S, C)
    X = batch_data['pred_X'][mol_idx]
    K = torch.einsum('ij, jk, kl -> il', C_inv, X, C_T_inv)
    D_refer = cal_D_from_C(C, n_orb)

    ## another way to remove redundant
    P = torch.einsum('ij,jk->ik', D_refer / 2, S)
    Q = (torch.eye(P.shape[0], device=P.device) - P).type(D_refer.dtype)
    K_no_redundant = None
    X_no_redundant = torch.einsum('ik,kj,lj -> il', P, X, Q) + torch.einsum('ik,kj,lj -> il', Q, X, P)

    # mask_lu = torch.zeros_like(K)
    # mask_lu[:n_orb, :n_orb] = 1
    # mask_rb = torch.zeros_like(K)
    # mask_rb[n_orb:, n_orb:] = 1

    # mask_lb = torch.zeros_like(K)
    # mask_lb[:n_orb, n_orb:] = 1
    # mask_ru = mask_lb.transpose(-1, -2)
    # K_no_redundant = mask_lb * K + mask_ru * K

    # X_no_redundant = torch.einsum('ik,kj,lj -> il', C, K_no_redundant, C)

    U_dagger = torch.torch.linalg.matrix_exp(-torch.einsum('ik,kj->ij', X_no_redundant, S))
    U = torch.linalg.matrix_exp(torch.einsum('ik,kj->ij', S, X_no_redundant))
    D_pred = torch.einsum('ij,jk,kl->il', U_dagger, D_refer, U)
    
    return D_pred, K_no_redundant
