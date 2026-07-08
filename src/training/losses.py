import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import mse_loss, l1_loss,huber_loss
from collections import defaultdict
from transformers import get_polynomial_decay_schedule_with_warmup
import glob
import os
import numpy as np
from pytorch_lightning import LightningModule
from ..models.model import create_model
from ..utility.pyscf import get_pyscf_obj_from_dataset, get_homo_lumo_from_h, get_energy_from_h
from ..dataset.buildblock import get_conv_variable_lin,block2matrix
from ..utility.eigen_solver import ED_trunc_p, ED_trunc, ED_PI_Layer
from functools import partial
import torch_geometric.transforms as T
import random
import pickle
import lmdb


HATREE_TO_KCAL = 627.5096

class SCFMonitor():
    def __init__(self, mol, ori_get_veff):
        self.mol = mol
        self.call_count = 0
        self.ori_get_veff = ori_get_veff

    def shadowed_get_veff(self, *args, **kwargs):
        self.call_count += 1
        return self.ori_get_veff(*args, **kwargs)
    
class ErrorMetric():
    def __init__(self,loss_weight):
        # if loss_weight == 0:
        #     raise ValueError(f"loss weight is 0, please check your each loss weight")
        pass
    def get_loss_from_diff(self, diff,metric,norm_nmae=None,norm_rmae=None):
        if metric == "mae":
            loss  =  torch.mean(torch.abs(diff))
        elif metric == "nmae":
            if norm_nmae is not None:
                loss  =  torch.mean(torch.abs(diff / norm_nmae))
            else:
                raise ValueError(f"norm_nmae is needed for {metric}")
        elif metric == "rmae":
            if norm_rmae is not None:
                loss  =  torch.mean(torch.abs(diff / norm_rmae))
            else:
                raise ValueError(f"norm_rmae is needed for {metric}")
        elif (metric == "nmaermae") or (metric == "rmaenmae"):
            if (norm_rmae is not None) and (norm_nmae is not None):
                nmae = torch.mean(torch.abs(diff / norm_nmae))
                rmae = torch.mean(torch.abs(diff / norm_rmae)) 
                loss = nmae + rmae
            else:
                raise ValueError(f"norm is needed for {metric}")
        elif metric == "ae":
            loss  =  torch.sum(torch.abs(diff))
        elif metric == "mse":
            loss =  torch.mean(diff**2)
        elif metric == "se":
            loss =  torch.sum(diff**2)
        elif metric == "rmse":
            loss  = torch.sqrt(torch.mean(diff**2))
        elif (metric == "maemse") or (metric == "msemae"):
            mae = torch.mean(torch.abs(diff))
            mse = torch.mean(diff**2)
            loss =  mae+mse
        elif (metric == "maemse") or (metric == "msemae"):
            mae = torch.mean(torch.abs(diff))
            mse = torch.mean(diff**2)
            loss =  mae+mse
        elif metric == 'huber':
            loss = huber_loss(diff, 0, reduction="mean", delta=1.0)
        else:
            raise ValueError(f"loss not support metric: {metric}")
        return loss
    
    def cal_loss(self,batch_data,error_dict = {},metric = None):
        pass

class ExceedPiError(ErrorMetric):
    def __init__(self, loss_weight):
        super().__init__(loss_weight)
        self.metric = "mse"
        self.loss_weight = loss_weight
        self.name = "exceed_pi_error"
    def cal_loss(self, batch_data,error_dict = {}):
        error_dict["loss"] = error_dict.get("loss",0)
        K_pred_list = batch_data['K_pred']
        loss = 0
        for K in K_pred_list:
            diff = torch.clamp(K - torch.pi, min=0) + torch.clamp(-torch.pi - K, min=0)
            loss += self.get_loss_from_diff(diff,self.metric)

        loss = loss/len(K_pred_list)
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'exceed_pi_error_{self.metric}'] = loss.detach()

class DFTAccRatio():
    def __init__(self, xc, acc_path, pos_unit="angstrom", basis="def2-svp"):
        self.xc = xc
        self.acc_path = acc_path
        self.name = "acc_ratio"
        self.pos_unit = pos_unit
        self.basis = basis

        os.makedirs(self.acc_path, exist_ok=True)
        self.path_num_cycles_minao = os.path.join(self.acc_path, 'num_cycles_minao.lmdb')
        self.env = lmdb.open(self.path_num_cycles_minao, map_size=10**9)
    
    def get_cycle_pred(self, single_mol, dm_pred):
        from gpu4pyscf.dft import rks
        mf_rks = rks.RKS(single_mol, xc=self.xc)
        monitor_mol = SCFMonitor(single_mol, mf_rks.get_veff)
        mf_rks.get_veff = monitor_mol.shadowed_get_veff
        mf_rks.kernel(dm0=dm_pred)
        cycle_pred = monitor_mol.call_count
        return cycle_pred
    
    def get_cycle_minao(self, single_mol):
        from gpu4pyscf.dft import rks
        mf_rks = rks.RKS(single_mol, xc=self.xc)
        monitor_mol = SCFMonitor(single_mol, mf_rks.get_veff)
        mf_rks.get_veff = monitor_mol.shadowed_get_veff
        dm_minao = mf_rks.get_init_guess(single_mol, "minao")
        mf_rks.kernel(dm0=dm_minao)
        cycle_minao = monitor_mol.call_count
        return cycle_minao

    def cal_loss(self, batch_data, error_dict={}):
        import pyscf
        # from pyscf.dft import rks
        import cupy as cp
        error_dict["loss"] = error_dict.get("loss",0)
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']

        acc_ratio = 0

        start_atom_idx = 0
        for i in range(len(D_pred_list)):
            D_pred = cp.asarray(D_pred_list[i]).astype(cp.float64)
            
            mol_size = batch_data["molecule_size"][i].detach().cpu().numpy()
            atoms = batch_data["atomic_numbers"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            pos = batch_data["pos"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            start_atom_idx = start_atom_idx + mol_size
            single_mol = pyscf.gto.Mole()
            single_mol.atom = [[atoms[j], pos[j]] for j in range(len(atoms))]
            single_mol.unit = self.pos_unit
            single_mol.basis = self.basis
            single_mol.build()
            mol_id = batch_data["idx"][i]
            
            cycle_pred = self.get_cycle_pred(single_mol, D_pred)
            with self.env.begin(write=True) as txn:
                raw_data = txn.get(f"mol_{mol_id}".encode('ascii'))
                if raw_data is None:
                    print(f"mol {mol_id} is not in lmdb, generating")
                    cycle_minao = self.get_cycle_minao(single_mol)
                    data_dict = {"mol_id": mol_id, "cycles": cycle_minao}
                    txn.put(f'mol_{mol_id}'.encode('ascii'), pickle.dumps(data_dict))
                else:
                    data_dict = pickle.loads(raw_data)
                    cycle_minao = data_dict["cycles"]
            acc_ratio += torch.tensor(cycle_pred / cycle_minao).to(D_pred_list[0].device)
        
        acc_ratio = acc_ratio/len(D_pred_list)
        error_dict[f'acc_ratio'] = acc_ratio

        

class TotalEnergyError(ErrorMetric):
    def __init__(self, xc, pos_unit = "angstrom", basis = "def2-svp"):
        super().__init__(1)
        self.metric = "mae"
        self.loss_weight = 1
        self.xc = xc
        self.pos_unit = pos_unit
        self.basis = basis
        self.name = "total_energy_loss"
    
    def cal_loss(self, batch_data,error_dict = {}):
        # TODO: recheck
        from gpu4pyscf.dft import rks
        import pyscf
        # from pyscf.dft import rks
        import cupy as cp
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        overlap_list = batch_data['overlap']
        loss = 0

        start_atom_idx = 0
        for i in range(len(D_pred_list)):
            D_pred = cp.asarray(D_pred_list[i]).astype(cp.float64)
            D_gt = cp.asarray(D_gt_list[i]).astype(cp.float64)
            
            mol_size = batch_data["molecule_size"][i].detach().cpu().numpy()
            atoms = batch_data["atomic_numbers"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            pos = batch_data["pos"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            start_atom_idx = start_atom_idx + mol_size
            single_mol = pyscf.gto.Mole()
            single_mol.atom = [[atoms[j], pos[j]] for j in range(len(atoms))]
            single_mol.unit = self.pos_unit
            single_mol.basis = self.basis
            single_mol.build()

            mf_rks = rks.RKS(single_mol, xc=self.xc)

            energy_pred = mf_rks.energy_tot(dm=D_pred)
            energy_gt = mf_rks.energy_tot(dm=D_gt)
            

            diff = torch.tensor(energy_pred - energy_gt).to(D_pred_list[0].device)
            # print(diff)
            # print(cp.mean(cp.abs(D_gt - D_pred)))
            loss += self.get_loss_from_diff(diff,self.metric)
        
        loss = loss/len(D_pred_list)
        # energy loss is only used in test
        error_dict[f'total_energy_loss_{self.metric}'] = loss.detach()

class DMbasedHamiltonianError(ErrorMetric):
    def __init__(self, xc, pos_unit = "angstrom", basis = "def2-svp"):
        super().__init__(1)
        self.metric = "mae"
        self.loss_weight = 1
        self.pos_unit = pos_unit
        self.xc = xc
        self.basis = basis
        self.name = "density_mat_based_hamiltonian_loss"
    
    def cal_loss(self, batch_data,error_dict = {}):
        # TODO: recheck
        from gpu4pyscf.dft import rks
        import pyscf
        import cupy as cp
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        H_gt_list = batch_data['fock']
        loss = 0

        start_atom_idx = 0
        for i in range(len(D_pred_list)):
            D_pred = cp.asarray(D_pred_list[i]).astype(cp.float64)
            H_gt = H_gt_list[i]
            
            mol_size = batch_data["molecule_size"][i].detach().cpu().numpy()
            atoms = batch_data["atomic_numbers"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            pos = batch_data["pos"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            start_atom_idx = start_atom_idx + mol_size
            single_mol = pyscf.gto.Mole()
            single_mol.atom = [[atoms[j], pos[j]] for j in range(len(atoms))]
            single_mol.unit = self.pos_unit
            single_mol.basis = self.basis
            single_mol.build()

            mf_rks = rks.RKS(single_mol, xc=self.xc)

            hamiltonian_pred = torch.tensor(mf_rks.get_fock(dm=D_pred).get()).to(D_pred_list[0].device)
            
            diff = hamiltonian_pred - H_gt
            
            loss += self.get_loss_from_diff(diff,self.metric)
        
        loss = loss/len(D_pred_list)
        # DM based hamiltonian loss is only used in test
        error_dict[f'density_mat_based_hamiltonian_loss_{self.metric}'] = loss.detach()


class RealSpaceRhoError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae", pos_unit = "angstrom", basis = "def2-svp"):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.pos_unit = pos_unit
        self.basis = basis
        self.name = "real_space_rho_loss"

    def cal_loss(self, batch_data,error_dict = {},metric = None):
        import pyscf
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        loss = 0
        for i in range(len(D_pred_list)):
            
            D_pred = D_pred_list[i]
            D_gt = D_gt_list[i]

            single_mol = pyscf.gto.Mole()
            single_mol.atom = batch_data["atom_pyscf"][i]
            single_mol.basis = self.basis
            single_mol.unit = self.pos_unit
            single_mol.verbose = 0
            single_mol.build(dump_input=False, parse_arg=False)

            coords = batch_data["grid_coords"][i]
            task = (coords, single_mol)
            ao_values = self._gen_ao(task, 0)

            c0_pred = torch.einsum('gi,ij->gj', ao_values, D_pred)
            rho_pred = torch.einsum('gj,gj->g', c0_pred, ao_values) * batch_data["grid_weights"][i]

            c0_gt = torch.einsum('gi,ij->gj', ao_values, D_gt)
            rho_gt = torch.einsum('gj,gj->g', c0_gt, ao_values) * batch_data["grid_weights"][i] 

            diff = torch.sum(torch.abs(rho_gt - rho_pred)) / torch.sum(torch.abs(rho_gt))
            loss += diff
        
        # Mole is necessary for rho calculation, it must be deleted to avoid out of memory
        # self._delete_mol(batch_data)
        loss = loss/len(D_pred_list)
        if metric in ["msemae","maemse"]:
            error_dict[f'real_space_rho_loss_mae'] = torch.mean(torch.abs(diff.detach()))
            error_dict[f'real_space_rho_loss_mse'] = torch.mean((diff.detach())**2)
        
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'real_space_rho_loss_{metric}'] = loss.detach()
    
    def _gen_ao(self, item, deriv = 0):
        from .eval_gto import eval_gto
        import cupy
        coord, mol = item
        if deriv == 0:
            ao_value = eval_gto(mol, "GTOval_sph_deriv0", coord)
        elif deriv == 1: 
            ao_value = eval_gto(mol, "GTOval_sph_deriv1", coord)

        # ao_value = cupy.expand_dims(ao_value, 0) #cupy.array (cuda)
        ao_value = torch.as_tensor(ao_value, device=coord.device).type(coord.dtype)
        return ao_value

    def _delete_mol(self, batch_data):
        if "single_mol" in batch_data.keys():
            mols = batch_data["single_mol"]
            for mol in mols:
                mol.__dict__.clear()
            batch_data["single_mol"] = None
            del batch_data["single_mol"]
        return

class EnergyAlignDMError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae"):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "energy_align_dm_loss"

    def cal_loss(self, batch_data,error_dict = {},metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        H_gt_list = batch_data['fock']
        overlap_list = batch_data['overlap']
        loss = 0
        for i in range(len(D_pred_list)):
            # F_D_S = torch.einsum('ij,jk,kl->il', H_gt_list[i], D_gt_list[i], overlap_list[i])
            F_D_S = torch.einsum('ij,jk,kl->il', H_gt_list[i], D_pred_list[i], overlap_list[i])
            S_D_F = torch.einsum('ij,jk,kl->il', overlap_list[i], D_pred_list[i], H_gt_list[i])
            diff = F_D_S - S_D_F
            loss += self.get_loss_from_diff(diff,metric)
        
        loss = loss/len(D_pred_list)
        if metric in ["msemae","maemse"]:
            error_dict[f'energy_align_dm_loss_mae'] = torch.mean(torch.abs(diff.detach()))
            error_dict[f'energy_align_dm_loss_mse'] = torch.mean((diff.detach())**2)
        
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'energy_align_dm_loss_{metric}'] = loss.detach()

class HamWeightedDMError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae"):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "hamiltonian_weighted_dm_loss"

    def cal_loss(self, batch_data,error_dict = {},metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        H_gt_list = batch_data['fock']
        loss = 0
        for i in range(len(D_pred_list)):
            norm_nmae = None
            norm_rmae = None
            trHD_pred = torch.sum(H_gt_list[i] * D_pred_list[i])
            trHD_gt = torch.sum(H_gt_list[i] * D_gt_list[i])
            diff = trHD_gt - trHD_pred
            if "nmae" in self.metric:
                norm_nmae = torch.sum(torch.abs(H_gt_list[i] * (D_gt_list[i] - D_pred_list[i].detach())))
            if "rmae" in self.metric:
                norm_rmae = trHD_gt
            loss += self.get_loss_from_diff(diff,metric,norm_nmae,norm_rmae)
        
        loss = loss/len(D_pred_list)
        if metric in ["msemae","maemse"]:
            error_dict[f'hamiltonian_weighted_dm_loss_mae'] = torch.mean(torch.abs(diff.detach()))
            error_dict[f'hamiltonian_weighted_dm_loss_mse'] = torch.mean((diff.detach())**2)
        if metric in ["nmaermae","rmaenmae"]:
            error_dict[f'hamiltonian_weighted_dm_loss_mae'] = torch.mean(torch.abs(diff.detach()))
            error_dict[f'hamiltonian_weighted_dm_loss_nmae'] = torch.mean(torch.abs(diff.detach() / norm_nmae))
            error_dict[f'hamiltonian_weighted_dm_loss_rmae'] = torch.mean(torch.abs(diff.detach() / norm_rmae))
        
        
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'hamiltonian_weighted_dm_loss_{metric}'] = loss.detach()

class EcoreError(ErrorMetric):
    def __init__(self, loss_weight=1, metric = "mae", pos_unit = "angstrom", basis = "def2-svp"):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.pos_unit = pos_unit
        self.basis = basis
        self.name = "Ecore_error"

    def cal_loss(self, batch_data,error_dict = {},metric = None):
        import pyscf
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        loss = 0
        start_atom_idx = 0
        for i in range(len(D_pred_list)):
            mol_size = batch_data["molecule_size"][i].detach().cpu().numpy()
            atoms = batch_data["atomic_numbers"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            pos = batch_data["pos"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            start_atom_idx = start_atom_idx + mol_size
            single_mol = pyscf.gto.Mole()
            single_mol.atom = [[atoms[j], pos[j]] for j in range(len(atoms))]
            single_mol.unit = self.pos_unit
            single_mol.basis = self.basis
            single_mol.build()
            h_core = single_mol.intor('int1e_kin') + single_mol.intor('int1e_nuc')
            D_pred = D_pred_list[i]
            D_gt = D_gt_list[i]
            h_core = torch.from_numpy(h_core).type(D_pred.dtype).to(D_pred.device)
            Ecore_pred = torch.sum(D_pred * h_core)
            Ecore_gt = torch.sum(D_gt * h_core)
            diff = Ecore_pred - Ecore_gt
            loss += self.get_loss_from_diff(diff,metric)
        
        loss = loss/len(D_pred_list)
        if metric in ["msemae","maemse"]:
            error_dict[f'ecore_loss_mae'] = torch.mean(torch.abs(diff.detach()))
            error_dict[f'ecore_loss_mse'] = torch.mean((diff.detach())**2)
        
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'ecore_loss_{metric}'] = loss.detach()

class DipoleError(ErrorMetric):
    def __init__(self, loss_weight=1, metric = "mae", pos_unit = "angstrom", basis = "def2-svp"):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.pos_unit = pos_unit
        self.basis = basis
        self.name = "dipole_error"

    def cal_loss(self, batch_data,error_dict = {},metric = None):
        import pyscf
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        loss = 0
        start_atom_idx = 0
        for i in range(len(D_pred_list)):
            mol_size = batch_data["molecule_size"][i].detach().cpu().numpy()
            atoms = batch_data["atomic_numbers"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            pos = batch_data["pos"][start_atom_idx:start_atom_idx + mol_size].detach().cpu().numpy()
            start_atom_idx = start_atom_idx + mol_size
            single_mol = pyscf.gto.Mole()
            single_mol.atom = [[atoms[j], pos[j]] for j in range(len(atoms))]
            single_mol.unit = self.pos_unit
            single_mol.basis = self.basis
            single_mol.build()

            D_pred = D_pred_list[i]
            D_gt = D_gt_list[i]

            ao_dip = torch.tensor(single_mol.intor_symmetric("int1e_r")).to(D_pred.device).type(D_pred.dtype)
            dip_moment_gt = -torch.einsum("uv, xvu->x", D_gt, ao_dip) + \
             torch.tensor(np.einsum("i, ix->x", single_mol.atom_charges(), 
                                    single_mol.atom_coords())).to(D_pred.device).type(D_pred.dtype) * 2.541746
            dip_moment_pred = -torch.einsum("uv, xvu->x", D_pred, ao_dip) + \
             torch.tensor(np.einsum("i, ix->x", single_mol.atom_charges(), 
                                    single_mol.atom_coords())).to(D_pred.device).type(D_pred.dtype) * 2.541746

            diff = dip_moment_pred - dip_moment_gt
            loss += self.get_loss_from_diff(diff,metric)
        
        loss = loss/len(D_pred_list)
        if metric in ["msemae","maemse"]:
            error_dict[f'dipole_loss_mae'] = torch.mean(torch.abs(diff.detach()))
            error_dict[f'dipole_loss_mae_loss_mse'] = torch.mean((diff.detach())**2)
        
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'dipole_loss_{metric}'] = loss.detach()

class OrthDensityMatrixError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae"):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "orthogonalized_density_matrix_loss"

    def cal_loss(self, batch_data,error_dict = {},metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        overlap_list = batch_data['overlap']
        loss = 0
        for i in range(len(D_pred_list)):
            S_half = self._get_S_half(overlap_list[i])
            diff = D_pred_list[i] - D_gt_list[i]
            diff = torch.einsum('ik,kl,lj->ij', S_half, diff, S_half)
            loss += self.get_loss_from_diff(diff,metric)
        
        loss = loss/len(D_pred_list)
        if metric in ["msemae","maemse"]:
            error_dict[f'Ortho_DM_loss_mae'] = torch.mean(torch.abs(diff.detach()))
            error_dict[f'Ortho_DM_loss_mse'] = torch.mean((diff.detach())**2)
        
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'Ortho_DM_loss_{metric}'] = loss.detach()
    
    def _get_S_half(self, S, epsilon=1e-12):
        L, V = torch.linalg.eigh(S)
        L_sqrt = torch.diag(torch.sqrt(torch.clamp(L, min=epsilon)))
        S_half = V @ L_sqrt @ V.t()
        return S_half

class DensityMatrixError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae"):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "density_matrix_loss"

    def cal_loss(self, batch_data,error_dict = {},metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        D_pred_list = batch_data['D_pred']
        D_gt_list = batch_data['D_gt']
        loss = 0
        for i in range(len(D_pred_list)):
            diff = D_pred_list[i] - D_gt_list[i]
            loss += self.get_loss_from_diff(diff,metric)
        
        loss = loss/len(D_pred_list)
        if metric in ["msemae","maemse"]:
            error_dict[f'DM_loss_mae'] = torch.mean(torch.abs(diff.detach()))
            error_dict[f'DM_loss_mse'] = torch.mean((diff.detach())**2)
        
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'DM_loss_{metric}'] = loss.detach()
        
class HamiltonianError(ErrorMetric):
    def __init__(self, loss_weight, metric = "mae", sparse = False, sparse_coeff = 1e-5, hami_name = 'HamiHead'):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "hamiltonian_loss"
        self.sparse = sparse
        self.sparse_coeff = sparse_coeff
        self.symmetry = 'symmetry' in hami_name.lower()


    def cal_loss(self, batch_data,error_dict = {},metric = None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        
        diag_mask = batch_data['diag_mask']
        non_diag_mask = batch_data['non_diag_mask']
        pred_diag = batch_data['pred_hamiltonian_diagonal_blocks']
        pred_non_diag = batch_data['pred_hamiltonian_non_diagonal_blocks']
        target_diag = batch_data['diag_hamiltonian']
        target_non_diag = batch_data['non_diag_hamiltonian']

        if self.symmetry:
            mask = torch.cat((diag_mask, non_diag_mask, non_diag_mask))
            predict = torch.cat((pred_diag, pred_non_diag, pred_non_diag))
            target = torch.cat((target_diag, target_non_diag, target_non_diag))
        else:
            mask = torch.cat((diag_mask, non_diag_mask))
            predict = torch.cat((pred_diag, pred_non_diag))
            target = torch.cat((target_diag, target_non_diag))

        if self.sparse:
            # target geq to sparse coeff is considered as non-zero
            sparse_mask = torch.abs(target).ge(self.sparse_coeff).float()
            target = target*sparse_mask
        diff = (predict-target)*mask
        
        weight = (mask.numel() / mask.sum())
        if metric == 'multi_head_mae':
            error_dict[f'hami_loss_mae'] = weight * torch.mean(torch.abs(diff)).detach()
            indices_list = batch_data['multi_head_indices']
            weight_list = [1,2,6,20]
            loss = 0
            for i in range(len(indices_list)):
                diff = batch_data['non_diag_hamiltonian'][indices_list[i]] - batch_data['pred_hamiltonian_non_diagonal_blocks'][indices_list[i]]
                mae = weight_list[i] * torch.mean(torch.abs(diff))
                error_dict[f'hami_loss_mae_non_diag_{i}'] = mae.detach()
                loss += mae
            diag_diff = batch_data['diag_hamiltonian'] - batch_data['pred_hamiltonian_diagonal_blocks']
            diag_mae = torch.mean(torch.abs(diag_diff))
            error_dict[f'hami_loss_mae_diag'] = diag_mae.detach()
            loss += diag_mae
        else:
            loss = self.get_loss_from_diff(diff,metric)
        if metric == "rmse":
            loss = loss*weight**0.5
        else:
            loss = loss*weight
        if metric in ["msemae","maemse"]:
            error_dict[f'hami_loss_mae'] = weight*torch.mean(torch.abs(diff.detach()))
            error_dict[f'hami_loss_mse'] = weight*torch.mean((diff.detach())**2)
            
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'hami_loss_{metric}'] = loss.detach()
        # print(f"==============hami_loss_{metric}, {loss.detach()}")

class EnergyHamiError(ErrorMetric):
    def __init__(self, loss_weight, trainer = None,metric="mae", 
                    basis="def2-svp", transform_h=False, scaled=False, normalization=False):
        super().__init__(loss_weight)
        
        self.trainer = trainer
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "energy_hami_loss"
        self.basis = basis
        self.transform_h = transform_h
        self.scaled = scaled
        self.normalization = normalization
        if normalization:
            self.mean_diag = torch.zeros(37, 37)
            self.mean_non_diag = torch.zeros(37, 37)
            self.std_diag = torch.load('std_diag.pt')
            self.std_non_diag = torch.load('std_non_diag.pt')
    
    def _batch_energy_hami(self, batch_data):
        batch_size = batch_data['idx'].shape[0]
        energy = batch_data['energy'] if 'energy' in batch_data.keys() else batch_data['idx']
        if self.normalization:
            batch_data['pred_hamiltonian_diagonal_blocks'] = \
                batch_data['pred_hamiltonian_diagonal_blocks'] * \
                    self.std_diag[None, :, :].to(batch_data['pred_hamiltonian_diagonal_blocks'].device) + \
                    self.mean_diag[None, :, :].to(batch_data['pred_hamiltonian_diagonal_blocks'].device)
            batch_data['pred_hamiltonian_non_diagonal_blocks'] = \
                batch_data['pred_hamiltonian_non_diagonal_blocks'] * \
                self.std_non_diag[None, :, :].to(batch_data['pred_hamiltonian_non_diagonal_blocks'].device) + \
                self.mean_non_diag[None, :, :].to(batch_data['pred_hamiltonian_non_diagonal_blocks'].device)
        elif self.scaled:
            diag_mask, non_diag_mask = batch_data['diag_mask'], batch_data['non_diag_mask']
            diag_target, non_diag_target = batch_data['diag_hamiltonian'], batch_data['non_diag_hamiltonian']

            sample_weight = diag_mask.size(1) * diag_mask.size(2) / diag_mask.sum(axis=(1,2))
            mean_target = diag_target.abs().mean(axis=(1,2)) * sample_weight
            batch_data['pred_hamiltonian_diagonal_blocks'] = batch_data['pred_hamiltonian_diagonal_blocks'] * mean_target[:, None, None]
            sample_weight = non_diag_mask.size(1) * non_diag_mask.size(2) / non_diag_mask.sum(axis=(1,2))
            mean_target = non_diag_target.abs().mean(axis=(1,2)) * sample_weight
            batch_data['pred_hamiltonian_non_diagonal_blocks'] = mean_target[:, None, None] * batch_data['pred_hamiltonian_non_diagonal_blocks']
        
        if 'pred_hamiltonian' not in batch_data:
            self.trainer.model.hami_model.build_final_matrix(batch_data)
        full_hami = batch_data['pred_hamiltonian']
        hami_energy = torch.zeros_like(energy,dtype=torch.float64)
        target_energy = torch.zeros_like(energy,dtype=torch.float64)
        hami_humo_lumo = torch.zeros_like(energy,dtype=torch.float64)
        target_humo_lumo = torch.zeros_like(energy,dtype=torch.float64)
        hami_coeff = torch.zeros_like(energy,dtype=torch.float64)

        target_hami = batch_data["hamiltonian"]

        for i in range(batch_size):
            start , end = batch_data['ptr'][i],batch_data['ptr'][i+1]
            pos = batch_data['pos'][start:end].detach().cpu().numpy()
            atomic_numbers = batch_data['atomic_numbers'][start:end].detach().cpu().numpy()
            mol, mf,factory = get_pyscf_obj_from_dataset(pos,atomic_numbers, basis=self.basis, 
                                                         xc='b3lyp5', gpu=False, verbose=1)
            dm0 = mf.init_guess_by_minao()
            init_h = mf.get_fock(dm=dm0)

            if self.trainer.hparams.remove_init:
                f_hi = full_hami[i].detach().cpu().numpy()+init_h
                f_gti = target_hami[i].detach().cpu().numpy()+init_h
            else:
                f_hi = full_hami[i].detach().cpu().numpy()
                f_gti = target_hami[i].detach().cpu().numpy()

            hami_energy[i] = get_energy_from_h(mf, f_hi)
            target_energy[i] = get_energy_from_h(mf, f_gti)

            hami_humo_lumo[i], hami_mo_coeff, mo_energy_pred = get_homo_lumo_from_h(mf, f_hi)
            target_humo_lumo[i], target_mo_coeff, mo_energy_target = get_homo_lumo_from_h(mf, f_gti)
            target_energy[i] = torch.mean(torch.abs(torch.tensor(mo_energy_pred - mo_energy_target)))

            hami_coeff[i] = torch.cosine_similarity(torch.tensor(hami_mo_coeff), torch.tensor(target_mo_coeff), dim=0).abs().mean()

            if factory is not None:factory.free_resources()

        return hami_energy, target_energy, hami_humo_lumo-target_humo_lumo, hami_coeff
    
    def cal_loss(self, batch_data, error_dict = {}, metric = None):
        metric = self.metric if metric is None else metric
        
        predict, target, humo_lumo_gap_diff, mo_coeff = self._batch_energy_hami(batch_data)
        error_dict['mo_coefficient'] = mo_coeff.mean()
        error_dict['energy_mae'] = torch.mean(target)

class OrbitalEnergyError(ErrorMetric):
    def __init__(self, loss_weight, metric="mae"):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "orbital_energy_loss"

    def cal_loss(self, batch_data, error_dict={}, metric=None):
        error_dict["loss"] = error_dict.get("loss",0)
        metric = self.metric if metric is None else metric
        obe_pred_list = batch_data['obe_pred']
        obe_gt_list = batch_data['obe_gt']
        n_orb = batch_data['n_orb']
        loss = 0
        for i in range(len(obe_pred_list)):
            diff = obe_pred_list[i][:n_orb[i]] - obe_gt_list[i][:n_orb[i]]
            loss += self.get_loss_from_diff(diff,metric)
        
        loss = loss/len(obe_pred_list)
        
        error_dict['loss']  += loss*self.loss_weight
        error_dict[f'orbital_energy_loss_{metric}'] = loss.detach()
        
        return error_dict
    
class _OrbitalEnergyErrorBase(ErrorMetric):
    @staticmethod
    def _iterate_batch(batch_data, basis):
        full_hami_pred = batch_data['pred_hamiltonian']
        full_hami = batch_data['fock']
        batch_size = batch_data['ptr'].shape[0] - 1

        for i in range(batch_size):
            start, end = batch_data['ptr'][i], batch_data['ptr'][i + 1]
            atomic_numbers = batch_data['atomic_numbers'][start:end]

            if 's1e' in batch_data:
                overlap_matrix = torch.from_numpy(batch_data['s1e'][i]).to(full_hami_pred[i].device)
            else:
                pos = batch_data['pos'][start:end].detach().cpu().numpy()
                mol, mf, factory = get_pyscf_obj_from_dataset(pos, atomic_numbers, basis=basis, gpu=True)
                s1e = mf.get_ovlp()
                overlap_matrix = torch.as_tensor(s1e, dtype=torch.float32, device=full_hami_pred[i].device)
                if factory: factory.free_resources()

            if 'fock_init' in batch_data:
                init_fock = batch_data['fock_init'][i]
                if isinstance(init_fock, np.ndarray):
                    init_fock = torch.from_numpy(init_fock)
                init_fock = init_fock.to(full_hami_pred[i].device)
                full_hami_pred_i = full_hami_pred[i] + init_fock
                full_hami_i = full_hami[i] + init_fock
            elif 'init_fock' in batch_data:
                init_fock = batch_data['init_fock'][i]
                if isinstance(init_fock, np.ndarray):
                    init_fock = torch.from_numpy(init_fock)
                init_fock = init_fock.to(full_hami_pred[i].device)
                full_hami_pred_i = full_hami_pred[i] + init_fock
                full_hami_i = full_hami[i] + init_fock
            else:
                full_hami_pred_i = full_hami_pred[i]
                full_hami_i = full_hami[i]

            yield atomic_numbers, overlap_matrix, full_hami_pred_i, full_hami_i, full_hami[i], full_hami_pred[i]


class GrassmannError(_OrbitalEnergyErrorBase):
    def __init__(self, enable_grassmann, enable_stationarity,
                 grassmann_weight, stationarity_weight, trainer=None,
                 basis="def2-svp", ed_type='trunc', trunc_factor=3.0,
                 grassmann_metric='projection', pi_iter=19):
        super().__init__(None)
        self.trainer = trainer
        self.enable_grassmann = enable_grassmann
        self.enable_stationarity = enable_stationarity
        self.grassmann_weight = grassmann_weight
        self.stationarity_weight = stationarity_weight
        self.basis = basis
        self.name = "grassmann_loss"
        self.ed_type = ed_type
        self.trunc_factor = trunc_factor
        self.grassmann_metric = grassmann_metric

    @staticmethod
    def _solve_eigh(full_hamiltonian, overlap_matrix, ed_type='trunc', trunc_factor=3.0):
        eps = 1e-8
        try:
            s_eigvals, s_eigvecs = torch.linalg.eigh(overlap_matrix)
            s_eigvals = torch.where(s_eigvals > eps, s_eigvals, eps)
            frac = s_eigvecs / torch.sqrt(s_eigvals).unsqueeze(-2)
            Fs = frac.T @ full_hamiltonian @ frac

            if ed_type == 'naive':
                e_vals, e_vecs = torch.linalg.eigh(Fs)
            elif ed_type == 'trunc':
                n = Fs.shape[-1]
                e_vals, e_vecs = ED_trunc.apply(Fs.unsqueeze(0), trunc_factor, n)
                e_vals, e_vecs = e_vals.squeeze(0), e_vecs.squeeze(0)
            elif ed_type == 'trunc_cpu':
                n = Fs.shape[-1]
                Fs_cpu = Fs.cpu()
                e_vals, e_vecs = ED_trunc.apply(Fs_cpu.unsqueeze(0), trunc_factor, n)
                e_vals = e_vals.to(Fs.device); e_vecs = e_vecs.to(Fs.device)
                e_vals, e_vecs = e_vals.squeeze(0), e_vecs.squeeze(0)
            elif ed_type == 'power_iteration':
                e_vals, e_vecs = ED_PI_Layer(Fs.unsqueeze(0), n, False)
                e_vals, e_vecs = e_vals.squeeze(0), e_vecs.squeeze(0)
            else:
                raise NotImplementedError(f"ed_type={ed_type}")

            e_vecs = frac @ e_vecs
            return True, e_vals, e_vecs
        except RuntimeError:
            return False, None, None

    def cal_loss(self, batch_data, error_dict={}, metric=None):
        if 'pred_hamiltonian' not in batch_data:
            self.trainer.model.hami_model.build_final_matrix(batch_data)
        error_dict["loss"] = error_dict.get("loss", 0)

        grass_losses, stat_losses = [], []
        compute_grass = self.enable_grassmann and self.grassmann_weight != 0
        compute_stat = self.enable_stationarity and self.stationarity_weight != 0
        if not (compute_grass or compute_stat):
            return error_dict

        batch_iterator = self._iterate_batch(batch_data, self.basis)
        for atomic_numbers, overlap_matrix, full_hami_pred_i, full_hami_i, _, _ in batch_iterator:
            nelec = atomic_numbers.sum().item()
            nocc = nelec // 2

            sym_gt, e_gt, c_gt = self._solve_eigh(full_hami_i, overlap_matrix, self.ed_type, self.trunc_factor)
            sym_pred, e_pred, c_pred = self._solve_eigh(full_hami_pred_i, overlap_matrix, self.ed_type, self.trunc_factor)

            if sym_gt and sym_pred:
                c_gt_occ = c_gt[:, :nocc]
                c_pred_occ = c_pred[:, :nocc]

                if compute_grass:
                    M_cross = c_gt_occ.T @ overlap_matrix @ c_pred_occ
                    if self.grassmann_metric == 'geodesic':
                        _, s, _ = torch.linalg.svd(M_cross)
                        theta = torch.acos(s.clamp(-1, 1))
                        grass_losses.append((theta ** 2).sum())
                    else:
                        grass_losses.append(nocc - torch.clamp((M_cross ** 2).sum(), max=nocc))

                if compute_stat:
                    D_pred = 2 * c_pred_occ @ c_pred_occ.T
                    delta = full_hami_i @ D_pred @ overlap_matrix - overlap_matrix @ D_pred @ full_hami_i
                    stat_losses.append((delta ** 2).sum())

        if compute_grass and grass_losses:
            lg = torch.stack(grass_losses).mean()
            error_dict['grassmann_loss'] = lg.detach()
            error_dict['loss'] += self.grassmann_weight * lg

        if compute_stat and stat_losses:
            ls = torch.stack(stat_losses).mean()
            error_dict['stationarity_loss'] = ls.detach()
            error_dict['loss'] += self.stationarity_weight * ls

        return error_dict
    