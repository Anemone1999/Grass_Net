import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import mse_loss, l1_loss,huber_loss
from collections import defaultdict
from transformers import get_polynomial_decay_schedule_with_warmup
import glob
import os

from pytorch_lightning import LightningModule
from ..models.model import create_model
from ..utility.pyscf import get_pyscf_obj_from_dataset, get_homo_lumo_from_h, get_energy_from_h
from ..dataset.buildblock import get_conv_variable_lin,block2matrix
from ..utility.eigen_solver import *
from functools import partial
import torch_geometric.transforms as T
import random
import time
import pickle

from .utils import cal_no_redundant_D, cal_D_from_H, check_fock, loss_analyse
from .losses import ExceedPiError, \
                    TotalEnergyError, \
                    EnergyAlignDMError, \
                    DensityMatrixError, \
                    HamiltonianError, \
                    EnergyHamiError, \
                    OrbitalEnergyError, \
                    HamWeightedDMError, \
                    RealSpaceRhoError, \
                    OrthDensityMatrixError, \
                    DFTAccRatio, \
                    DMbasedHamiltonianError, \
                    EcoreError, \
                    DipoleError, \
                    GrassmannError, \
                    GrassmannWarmupWrapper


class FloatCastDatasetWrapper(T.BaseTransform):
    """A transform that casts all floating point tensors to a given dtype.
    tensors to a given dtype.
    """

    def __init__(self, dtype=torch.float64):
        super(FloatCastDatasetWrapper, self).__init__()
        self._dtype = dtype

    def forward(self, data):
        for key, value in data:
            if torch.is_tensor(value) and torch.is_floating_point(value):
                setattr(data, key, value.to(self._dtype))
            if isinstance(value, list):
                if torch.is_tensor(value[0]) and torch.is_floating_point(value[0]):
                    setattr(data, key, [v.to(self._dtype) for v in value])
        return data


class LNNP(LightningModule):
    def __init__(self, hparams, mean=None, std=None):
        super(LNNP, self).__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.finetune_flag:
            ckpt_files = glob.glob(os.path.join(self.hparams.finetune_path, '*.ckpt'))  
            print(os.path.join(self.hparams.finetune_path, '*.ckpt'))
            if ckpt_files:  
                latest_file = max(ckpt_files, key=os.path.getctime)  
                print(f"The latest .ckpt file for finetuning is: {latest_file}")  
                print("initializing model from pretrained ckeckpoints")

                pretrained_ckpt = torch.load(latest_file, map_location='cpu')
                pretrain_hparams = pretrained_ckpt['hyper_parameters']
                pretrain_hparams.ckpt_path = self.hparams.finetune_ckpt_path
                pretrain_state_dict = pretrained_ckpt['state_dict']
                
                self.model = create_model(pretrain_hparams, mean, std)
                self.load_state_dict(pretrain_state_dict, strict=True)
            else: 
                print("No .ckpt files found in the finetune folder.")
        else:
            self.model = create_model(self.hparams, mean, std)

        self.enable_energy = self.hparams.enable_energy
        self.enable_forces = self.hparams.enable_forces
        self.enable_hami = self.hparams.enable_hami
        self.enable_hami_orbital_energy = self.hparams.enable_hami_orbital_energy
        self.enable_grassmann = self.hparams.get("enable_grassmann", False)
        self.enable_stationarity = self.hparams.get("enable_stationarity", False)
        self.enable_DM = self.hparams.enable_DM
        self.enable_exceed_pi = self.hparams.enable_exceed_pi
        self.enable_energy_align_DM = self.hparams.enable_energy_align_DM
        self.enable_trHD = self.hparams.enable_trHD
        self.enable_rho = self.hparams.enable_rho
        self.enable_ortho_DM = self.hparams.enable_ortho_DM

        self.enable_ecore = self.hparams.enable_ecore
        self.enable_dipole = self.hparams.enable_dipole

        self.enable_total_energy = self.hparams.enable_total_energy
        self.enable_acc_ratio = self.hparams.enable_acc_ratio
        self.enable_DM_based_H = self.hparams.enable_DM_based_H
        if self.hparams.pred_target == "H":
            self.construct_loss_func_list_H_pred()
        elif self.hparams.pred_target == "X":
            self.construct_loss_func_list_X_pred()
        else:
            raise ValueError(f"pred_target {self.hparams.pred_target} not supported, please choose from 'H' or 'X'")
        
        self._reset_losses_dict()

        dtype_mapping = {16: torch.float16, 32: torch.float, 64: torch.float64}
        self.data_transform = FloatCastDatasetWrapper(
            dtype_mapping[int(self.hparams.precision)]
        )

    def construct_loss_func_list_X_pred(self,):
        self.loss_func_list_train = []
        if self.enable_DM:
            self.loss_func_list_train.append(DensityMatrixError(self.hparams.dm_weight,
                                                                self.hparams.dm_train_loss))
        if self.enable_exceed_pi:
            self.loss_func_list_train.append(ExceedPiError(self.hparams.exceed_pi_weight))
        if self.enable_energy_align_DM:
            self.loss_func_list_train.append(EnergyAlignDMError(self.hparams.energy_align_dm_weight,
                                                                self.hparams.energy_align_dm_train_loss))
        if self.enable_trHD:
            self.loss_func_list_train.append(HamWeightedDMError(self.hparams.trHD_weight,
                                                                self.hparams.trHD_train_loss))
        if self.enable_rho:
            self.loss_func_list_train.append(RealSpaceRhoError(self.hparams.rho_weight,
                                                                self.hparams.rho_train_loss,
                                                                self.hparams.pos_unit,
                                                                self.hparams.basis))
        if self.enable_ortho_DM:
            self.loss_func_list_train.append(OrthDensityMatrixError(self.hparams.ortho_dm_weight,
                                                                    self.hparams.ortho_dm_train_loss))
        
        self.loss_func_list_val = []
        if self.enable_DM:
            self.loss_func_list_val.append(DensityMatrixError(self.hparams.dm_weight,
                                                              self.hparams.dm_val_loss))
        if self.enable_energy_align_DM:
            self.loss_func_list_val.append(EnergyAlignDMError(self.hparams.energy_align_dm_weight,
                                                              self.hparams.energy_align_dm_val_loss))
        if self.enable_trHD:
            self.loss_func_list_val.append(HamWeightedDMError(self.hparams.trHD_weight,
                                                              self.hparams.trHD_val_loss))
        if self.enable_rho:
            self.loss_func_list_val.append(RealSpaceRhoError(self.hparams.rho_weight,
                                                             self.hparams.rho_val_loss,
                                                             self.hparams.pos_unit,
                                                             self.hparams.basis))
        if self.enable_ortho_DM:
            self.loss_func_list_val.append(OrthDensityMatrixError(self.hparams.ortho_dm_weight,
                                                                    self.hparams.ortho_dm_val_loss))
        if self.enable_ecore:
            self.loss_func_list_val.append(EcoreError(pos_unit=self.hparams.pos_unit,
                                                      basis=self.hparams.basis))
        if self.enable_dipole:
            self.loss_func_list_val.append(DipoleError(pos_unit=self.hparams.pos_unit,
                                                       basis=self.hparams.basis))

        self.loss_func_list_test = self.loss_func_list_val[:]
        if self.enable_total_energy:
            self.loss_func_list_test.append(TotalEnergyError(self.hparams.xc_type, self.hparams.pos_unit, self.hparams.basis))
        if self.enable_acc_ratio:
            self.loss_func_list_test.append(DFTAccRatio(self.hparams.xc_type,
                                                        self.hparams.acc_path,
                                                        self.hparams.pos_unit,
                                                        self.hparams.basis))
        if self.enable_DM_based_H:
            self.loss_func_list_test.append(DMbasedHamiltonianError(self.hparams.xc_type, self.hparams.pos_unit, self.hparams.basis))
        self.loss_func_list_val_realworld = []

    def construct_loss_func_list_H_pred(self,):
        self.loss_func_list_train = []
        if self.enable_hami:
            self.loss_func_list_train.append(HamiltonianError(self.hparams.hami_weight,self.hparams.hami_train_loss, self.hparams.sparse_loss, self.hparams.sparse_loss_coeff, self.hparams.hami_model.name))
        if self.enable_hami_orbital_energy:
            self.loss_func_list_train.append(OrbitalEnergyError(self.hparams.orbital_energy_weight,
                 self.hparams.orbital_energy_train_loss))
        if self.enable_DM:
            self.loss_func_list_train.append(DensityMatrixError(self.hparams.dm_weight,self.hparams.dm_train_loss))
        if self.enable_energy_align_DM:
            self.loss_func_list_train.append(EnergyAlignDMError(self.hparams.energy_align_dm_weight,self.hparams.energy_align_dm_train_loss))
        if self.enable_rho:
            self.loss_func_list_train.append(RealSpaceRhoError(self.hparams.rho_weight,
                                                                self.hparams.rho_train_loss,
                                                                self.hparams.pos_unit,
                                                                self.hparams.basis))
        if self.enable_grassmann or self.enable_stationarity:
            grass_err = GrassmannError(
                self.enable_grassmann, self.enable_stationarity,
                self.hparams.grassmann_weight, self.hparams.stationarity_weight,
                self, self.hparams.basis, ed_type=self.hparams.ed_type,
                trunc_factor=self.hparams.get("ed_trunc_factor", 3.0),
                grassmann_metric=self.hparams.get("grassmann_metric", "projection"))

            warmup_steps = self.hparams.get("grassmann_warmup_steps", 0)
            if warmup_steps > 0:
                grass_err = GrassmannWarmupWrapper(grass_err, warmup_steps=warmup_steps)
            self.loss_func_list_train.append(grass_err)
        self.loss_func_list_val = []
        if self.enable_hami:
            self.loss_func_list_val.append(HamiltonianError(self.hparams.hami_weight,self.hparams.hami_val_loss, self.hparams.hami_model.name))
        if self.enable_hami_orbital_energy:
            self.loss_func_list_val.append(OrbitalEnergyError(self.hparams.orbital_energy_weight,
                 self.hparams.orbital_energy_val_loss))  
        if self.enable_DM:
            self.loss_func_list_val.append(DensityMatrixError(self.hparams.dm_weight,self.hparams.dm_val_loss))
        if self.enable_energy_align_DM:
            self.loss_func_list_val.append(EnergyAlignDMError(self.hparams.energy_align_dm_weight,self.hparams.energy_align_dm_val_loss)) 
        if self.enable_rho:
            self.loss_func_list_val.append(RealSpaceRhoError(self.hparams.rho_weight,
                                                             self.hparams.rho_val_loss,
                                                             self.hparams.pos_unit,
                                                             self.hparams.basis))
        if self.enable_grassmann or self.enable_stationarity:
            self.loss_func_list_val.append(GrassmannError(
                self.enable_grassmann, self.enable_stationarity,
                self.hparams.grassmann_weight, self.hparams.stationarity_weight,
                self, self.hparams.basis, ed_type=self.hparams.ed_type,
                trunc_factor=self.hparams.get("ed_trunc_factor", 3.0),
                grassmann_metric=self.hparams.get("grassmann_metric", "projection")))

        
        # some real world / application level evaluation.
        # a little time consuming, thus, in data module, only 1 batch data is used.
        self.loss_func_list_val_realworld = [] #self.loss_func_list_val[:]#
        if self.enable_hami and self.hparams.enable_energy_hami_error:
            self.loss_func_list_val_realworld.append(EnergyHamiError(1,
                                                                     self,
                                                                self.hparams.energy_val_loss, 
                                                                self.hparams.basis, 
                                                                "qh9" in self.hparams.data_name.lower(),
                                                                self.hparams.hami_train_loss=="scaled",
                                                                self.hparams.hami_train_loss== "normalization"))


        self.loss_func_list_test = self.loss_func_list_val[:]
        if self.enable_total_energy:
            self.loss_func_list_test.append(TotalEnergyError(self.hparams.xc_type,
                                                             self.hparams.pos_unit,
                                                             self.hparams.basis))
        if self.enable_acc_ratio:
            self.loss_func_list_test.append(DFTAccRatio(self.hparams.xc_type,
                                                        self.hparams.acc_path,
                                                        self.hparams.pos_unit,
                                                        self.hparams.basis))
        if self.enable_DM_based_H:
            self.loss_func_list_test.append(DMbasedHamiltonianError(self.hparams.xc_type, self.hparams.pos_unit, self.hparams.basis))

        if self.enable_hami and self.hparams.enable_energy_hami_error:
            self.loss_func_list_test.append(EnergyHamiError(1,
                                                            self,
                                                            self.hparams.energy_val_loss, 
                                                            self.hparams.basis, 
                                                            "qh9" in self.hparams.data_name.lower(),
                                                            self.hparams.hami_train_loss=="scaled",
                                                            self.hparams.hami_train_loss== "normalization"))
    

    def _reset_losses_dict(self,):
        self.losses = {"train":defaultdict(list),
                        "val":defaultdict(list),
                        "test":defaultdict(list)}
        
    def configure_optimizers(self):
        if not self.hparams.multi_para_group: 
            params = self.model.parameters()
        else:
            other_params = []
            pretrained_params = []
            hami_head = []
            hami_head_0 = []
            hami_head_1 = []
            hami_head_2 = []
            hami_head_3 = []
            for (name, param) in self.model.named_parameters():
                # load pretrain is not in key
                if self.hparams.model.load_pretrain != '':
                    if 'node_attr_encoder' in name: # in so2 model the node_attr_encoder is likely to be pretrained
                        pretrained_params.append(param)
                # elif 'LSRM_module' in name:
                #     pretrained_params.append(param)
                # elif 'e3_gnn_node_pair_layer' in name:
                #     pretrained_params.append(param)
                elif 'hami_model' in name:
                    if ('e3_gnn_node_pair_layer' in name) or ('fc_ij' in name) or ('expand_ij' in name):
                        if '_1.' in name:
                            hami_head_1.append(param)
                        elif '_2.' in name:
                            hami_head_2.append(param)
                        elif '_3.' in name:
                            hami_head_3.append(param)
                        else:
                            hami_head_0.append(param)
                    else:
                        hami_head.append(param)
                else:
                    other_params.append(param)
            params = [
                {'params': other_params},
                {'params': pretrained_params, 'lr': self.hparams.lr*0.5},
                {'params': hami_head, 'lr': self.hparams.lr*5},
                {'params': hami_head_0, 'lr': self.hparams.lr*5},
                {'params': hami_head_1, 'lr': self.hparams.lr*5},
                {'params': hami_head_2, 'lr': self.hparams.lr*5},
                {'params': hami_head_3, 'lr': self.hparams.lr*5},
            ]
        optimizer = AdamW(
            params,
            lr=self.hparams.lr,
            betas = (0.99,0.999),
            weight_decay=self.hparams.weight_decay,
            amsgrad=False
        )
        
        schedule_cfg = self.hparams["schedule"]
        #warm up is set in optimizer_step
        if schedule_cfg.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer,  schedule_cfg.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif schedule_cfg.lr_schedule == 'polynomial':
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=-1, 
                num_training_steps= self.hparams.max_steps,
                lr_end =  schedule_cfg.lr_min, power = 1.0, last_epoch = -1)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif schedule_cfg.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor= schedule_cfg.lr_factor,
                patience= schedule_cfg.lr_patience,
                min_lr= schedule_cfg.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown lr_schedule: {schedule_cfg.lr_schedule}")
        
        return [optimizer], [lr_scheduler]
    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        lr_warmup_steps = self.hparams["schedule"]["lr_warmup_steps"]
        if self.trainer.global_step < lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        super().optimizer_step(*args, **kwargs) 

        optimizer.zero_grad()
        
    def forward(self,  batch_data):
        return self.model(batch_data)

    def training_step(self, batch_data, batch_idx):
        if self.hparams.pred_target == "X":
            return self.step_X_pred(batch_data, "train", self.loss_func_list_train)
        elif self.hparams.pred_target == "H":
            return self.step(batch_data, "train", self.loss_func_list_train)

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm for diagnostics."""
        total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.parameters() if p.grad is not None) ** 0.5
        self.log('train/grad_norm', total_norm, on_step=True, prog_bar=False)

    def validation_step(self, batch_data, batch_idx, dataloader_idx=0):
        # validation step
        if dataloader_idx == 0:
            if self.hparams.pred_target == "X":
                return self.step_X_pred(batch_data, "val", self.loss_func_list_val)
            elif self.hparams.pred_target == "H":
                return self.step(batch_data, "val", self.loss_func_list_val)
        else:
            if self.loss_func_list_val_realworld:
                if self.hparams.pred_target == "X":
                    return self.step_X_pred(batch_data, "val", self.loss_func_list_val_realworld)
                elif self.hparams.pred_target == "H":
                    return self.step(batch_data, "val", self.loss_func_list_val_realworld)

    def test_step(self, batch_data, batch_idx):
        if self.enable_total_energy and (batch_idx >= self.hparams.max_sample_for_total_energy):
            # if total energy evaluation is enabled, we only evaluate it for few samples
            self.loss_func_list_test = self.loss_func_list_val[:]
        if self.hparams.pred_target == "X":
            return self.step_X_pred(batch_data, "test", self.loss_func_list_test)
        elif self.hparams.pred_target == "H":
            return self.step(batch_data, "test", self.loss_func_list_test)
    
    def _save_test_results(self, batch_data, error_dict):
        ### ############################################################# ###
        ### record errors for each single sample, so batch size must be 1 ###
        ### ############################################################# ###
        save_path = self.hparams.hydra_path
        if len(batch_data['cid']) != 1:
            # batch size > 1 is not supported in saving test results
            return
        saving_dict = {"cid": batch_data['cid'][0]}

        for key in error_dict:
            if key=='loss' and type(error_dict[key])==int:
                continue
            saving_dict[key] = error_dict[key].detach().cpu().numpy()
        
        with open(f"{save_path}/error_{self.hparams.data_name}.pkl", 'ab') as f:
            pickle.dump(saving_dict, f)
        return


    def step_X_pred(self, batch_data, stage, loss_func_list=[]):
        batch_data = self.data_transform(batch_data)
        with torch.set_grad_enabled(stage == "train" or self.enable_forces):
            batch_data = self(batch_data)
            batch_data = self.model.hami_model.build_final_matrix(batch_data, sym_type="asym")

        loss_func_list = self.loss_func_list_train if loss_func_list is [] else loss_func_list
        error_dict = {"loss":0}
            
        D_pred_list = []
        K_pred_list = []
        D_gt_list = []

        if ('D_gt' in batch_data.keys()) and (batch_data['D_gt'][0] is not None):
            for mol_idx in range(len(batch_data['idx'])):
                D_pred, K_pred = cal_no_redundant_D(batch_data, mol_idx)
                D_pred_list.append(D_pred)
                K_pred_list.append(K_pred)

            batch_data['D_pred'] = D_pred_list
            batch_data['K_pred'] = K_pred_list
        else:
            for mol_idx in range(len(batch_data['idx'])):
                D_pred, K_pred = cal_no_redundant_D(batch_data, mol_idx)
                D_gt = cal_D_from_H(batch_data, mol_idx, flag='gt', ed_type='naive')
                D_pred_list.append(D_pred)
                K_pred_list.append(K_pred)
                D_gt_list.append(D_gt)

            batch_data['D_pred'] = D_pred_list
            batch_data['K_pred'] = K_pred_list
            batch_data['D_gt'] = D_gt_list

        # check_fock(batch_data, self.hparams.xc_type, self.hparams.pos_unit, self.hparams.basis)
        # loss_analyse(batch_data)

        for loss_func in loss_func_list:
            loss_func.cal_loss(batch_data,error_dict)
        
        for key in error_dict:
            if key=='loss' and type(error_dict[key])==int:
                continue
            self.losses[stage][key].append(error_dict[key].detach())

        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {f"train_per_step/{k}": v for k, v in error_dict.items()}
            train_metrics['learningrate'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = int(self.trainer.global_step) 

            self.trainer.progress_bar_metrics["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.trainer.progress_bar_metrics["loss"] = error_dict["loss"].detach().item()

            self.log_dict(train_metrics, sync_dist=True)
        
        if stage == 'test':
            self._save_test_results(batch_data, error_dict)
            
        return error_dict["loss"]
        

    def step(self, batch_data, stage, loss_func_list=[]):
        batch_data = self.data_transform(batch_data)
        should_skip_local = torch.tensor([0], dtype=torch.int32, device=self.device)
        with torch.set_grad_enabled(stage == "train" or self.enable_forces):
            batch_data = self(batch_data)
        
        loss_func_list = self.loss_func_list_train if loss_func_list is [] else loss_func_list
        error_dict = {"loss":0}

        # Check whether active losses need D/obe (density matrix / orbital energy), which
        # requires expensive eigh via cal_D_from_H.  In training modes where only hamiltonian
        # and/or grassmann/stationarity losses are active, this is pure overhead.
        _active_loss_names = {getattr(lf, 'name', '') for lf in loss_func_list}
        _dm_obe_loss_names = {
            'density_matrix_loss', 'energy_align_dm_loss', 'orbital_energy_loss',
            'orthogonalized_density_matrix_loss', 'hamiltonian_weighted_dm_loss',
            'real_space_rho_loss', 'Ecore_error', 'dipole_error',
            'total_energy_loss', 'acc_ratio', 'density_mat_based_hamiltonian_loss'
        }
        _needs_dm_obe = bool(_active_loss_names & _dm_obe_loss_names)
        _needs_full_hami = bool(_active_loss_names & {'energy_hami_loss', 'grassmann_loss'}) or _needs_dm_obe

        if stage == "test" or (len(loss_func_list) > 1 and _needs_full_hami):
            batch_data = self.model.hami_model.build_final_matrix(batch_data, sym_type="sym")

        if (stage == "test" and _needs_dm_obe) or (stage == "train" and _needs_dm_obe):
            D_pred_list = []
            D_gt_list = []
            obe_pred_list = []
            obe_gt_list = []

            for i in range(len(batch_data["pred_hamiltonian"])):
                batch_data["fock"][i] = batch_data["fock"][i] + batch_data["fock_init"][i]
                batch_data["pred_hamiltonian"][i] = batch_data["pred_hamiltonian"][i] + batch_data["fock_init"][i]
                D_gt, obe_gt = cal_D_from_H(batch_data, i, flag='gt')
                D_gt_list.append(D_gt)
                obe_gt_list.append(obe_gt)
                D_pred, obe_pred = cal_D_from_H(batch_data, i, flag='pred', ed_type=self.hparams.ed_type)
                D_pred_list.append(D_pred)
                obe_pred_list.append(obe_pred)
                if D_gt is None or D_pred is None:
                    should_skip_local[0] = 1

            batch_data['D_gt'] = D_gt_list
            batch_data['D_pred'] = D_pred_list
            batch_data['obe_gt'] = obe_gt_list
            batch_data['obe_pred'] = obe_pred_list

            ## skip the batch since ed is failed
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(should_skip_local, op=torch.distributed.ReduceOp.MAX)
            if should_skip_local.item() == 1:
                if self.global_rank == 0:
                    print(f"skipping batch since ed failure")
                return None

        for loss_func in loss_func_list:
            loss_func.cal_loss(batch_data,error_dict)
            
        for key in error_dict:
            if key=='loss' and type(error_dict[key])==int:
                continue
            self.losses[stage][key].append(error_dict[key].detach())

        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {f"train_per_step/{k}": v for k, v in error_dict.items()}
            train_metrics['learningrate'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step 

            self.trainer.progress_bar_metrics["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.trainer.progress_bar_metrics["loss"] = error_dict["loss"].detach().item()
            
            self.log_dict(train_metrics, sync_dist=True)
        
        if stage == 'test':
            self._save_test_results(batch_data, error_dict)
        
        return error_dict["loss"]
    
    def _check_devices(self):
        self.model.representation_model.set()

    def on_fit_start(self) -> None:
        self._check_devices()

    def on_test_start(self) -> None:
        self._check_devices()

    def on_predict_start(self) -> None:
        self._check_devices()

    def on_train_epoch_end(self):
        dm = self.trainer.datamodule

    # TODO(shehzaidi): clean up this function, redundant logging if dy loss exists.
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {}

            for stage in ["train","val","test"]:
                for key in self.losses[stage]:
                    if stage == "val" and key == "loss":
                        result_dict["val_loss"] = torch.stack(self.losses[stage][key]).mean()                
                    result_dict[f"{stage}/{key}"] = torch.stack(self.losses[stage][key]).mean()
            self.log_dict(result_dict, sync_dist=True)
            print(result_dict)
        self._reset_losses_dict()
        
    def on_test_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {}

            for stage in ["train","val","test"]:
                for key in self.losses[stage]:
                    if stage == "val" and key == "loss":
                        result_dict["val_loss"] = torch.stack(self.losses[stage][key]).mean()                
                    else:
                        result_dict[f"{stage}/{key}"] = torch.stack(self.losses[stage][key]).mean()
            self.log_dict(result_dict, sync_dist=True)
            print(result_dict)
        self._reset_losses_dict()
