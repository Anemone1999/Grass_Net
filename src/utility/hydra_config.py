
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omegaconf import MISSING, DictConfig



@dataclass
class BaseSchema:
    """This is for CLI applications that need to reuse a CLI parameter in multiple places
    in the config file. The subfields of `general` are not fixed and can be anything. It
    also allows configs to be shared between multiple workstreams; e.g. the ks_config can
    be shared between data generation, training (as a callback) and evaluation.

    Examples:
        1) You can set `general.output_dir=foobar` and in other places in the config file
        `output_dir: ${general.output_dir}`.
        2) In the default field of the config, you can set

            ```
            default:
              - general/kohn_sham/default
            ```

        which is useful for composing configs.  This will append the default.yaml file from
        config_path/general/kohn_sham/ to the current config. In other places in the config
        file, you can then call ${general.kohn_sham}.
    """
    

@dataclass
class Config(BaseSchema):
    seed: int = 0
    hydra_path: str = "./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
    job_id: str = "auto"
    ckpt_path: str = '/data'
    save_path: str = '/data'
    log_dir: str = "./tmp"
    schedule: Dict[str, Any] = MISSING
    model: Dict[str, Any] = MISSING
    wandb: Dict[str, Any] = MISSING
    hydra: Dict[str, Any] = field(default_factory=lambda: {
        "run": {
            "dir": "./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
        }
    })
    #########
    # trainer related config
    finetune_flag: bool = False # distinguish whether the task is a finetune task
    finetune_path: str = "./none"
    finetune_ckpt_path: str = "./none"
    pred_target: str = "H" # H denotes Hamiltonian, X denotes rotation matrix
    model_backbone:str = "SPHNet"
    output_model: str = "EquivariantScalar_viaTP"
    hami_model: Dict[str, Any] = MISSING
    use_sparse_tp: bool = False
    sparsity: float = 0.7
    num_epochs: int = 300 #number of epochs
    max_steps: int = -1 #Maximum number of gradient steps.
    batch_size: int = 32 #batch size
    inference_batch_size: Any = None
    dataloader_num_workers: int = 4
    lr: float = 1e-4
    multi_para_group: bool = False
    weight_decay: float = 0
    enable_hami: bool = False
    enable_symmetry: bool = False
    enable_energy: bool = False
    enable_forces: bool = False
    enable_energy_hami_error: bool = False
    enable_hami_orbital_energy: bool = False
    enable_DM: bool = False
    enable_ortho_DM: bool = False
    enable_exceed_pi: bool = False
    enable_energy_align_DM: bool = False
    enable_trHD: bool = False
    enable_rho: bool = False
    enable_ecore: bool = False
    enable_dipole: bool = False
    enable_total_energy: bool = False
    enable_acc_ratio: bool = False
    enable_DM_based_H: bool = False
    max_sample_for_total_energy: int = 100
    energy_weight: float = 0 #Weighting factor for energies in the loss function
    forces_weight: float = 0 #Weighting factor for forces in the loss function
    hami_weight: float = 1 #Weighting factor for hami in the loss function
    dm_weight: float = 1 #Weighting factor for density matrix in the loss function
    ortho_dm_weight: float = 0 #Weighting factor for orthogonal density matrix in the loss function
    exceed_pi_weight: float = 0 #Weighting factor for exceed pi in the loss function
    energy_align_dm_weight: float = 0 #Weighting factor for energy aligned density matrix in the loss function
    trHD_weight: float = 0
    rho_weight: float = 0
    orbital_energy_weight: float = 0 #Weighting factor for orbital energy in the loss function
    xc_type: str = "b3lyp5"
    pos_unit: str = "angstrom"
    energy_train_loss: str = 'mse'
    forces_train_loss: str = 'mse'
    orbital_energy_train_loss: str = 'mae'
    orbital_energy_val_loss: str = 'mae'
    hami_train_loss: str = 'maemse'
    hami_val_loss: str = 'mae'
    dm_train_loss: str = 'mae'
    ortho_dm_train_loss: str = 'mae'
    dm_val_loss: str = 'mae'
    ortho_dm_val_loss: str = 'mae'
    energy_align_dm_train_loss: str = 'mae'
    energy_align_dm_val_loss: str = 'mae'
    trHD_train_loss: str = 'mae'
    trHD_val_loss: str = 'mae'
    rho_train_loss: str = 'mae'
    rho_val_loss: str = 'mae'
    energy_val_loss: str = 'mae'
    forces_val_loss: str = 'mae'
    enable_grassmann: bool = False
    enable_stationarity: bool = False
    grassmann_weight: float = 0.001
    stationarity_weight: float = 0.05
    grassmann_metric: str = 'projection'  # 'projection', 'densityS', or 'geodesic' (SVD, unstable)

    grassmann_pi_iter: int = 19
    ed_type: str = 'naive'
    ed_trunc_factor: float = 3.0
    sparse_loss: bool = False
    sparse_loss_coeff: float = 1e-3
    ngpus: int = 1
    num_nodes: int = 1
    gradient_clip_val: Any = None
    early_stopping_patience: int = 30
    val_check_interval: Any = None #follow pytorch lightning
    check_val_every_n_epoch: int = 5
    log_every_n_steps: int = 50
    skip_test: bool = False
    test_interval: int = 10 #Test interval, one test per n epochs (default = 10)
    save_interval: int = 10 #Save interval, one save per n epochs (default = 10)
    ############: Any
    #: Any data realted config
    basis: str = "def2-svp"  #when predict hamitonian, the basis need to be set
    data_name: str = "QH9"
    dataset_path: Any = None
    acc_path: Any = None
    index_path: Any  = None
    dataset_size: int  = -1 #the dataset size is used for debug. -1 is all data")
    train_ratio: Any = 0.8 # Percentage of samples in training set (null to use all remaining samples)
    val_ratio: Any = 0.02 # Percentage of samples in validation set (null to use all remaining samples)
    test_ratio: Any = 0.18 # Percentage of samples in test set (null to use all remaining samples)
    cutoff_lower: Any = 0.0 #Lower cutoff in model
    cutoff_upper: Any = 5.0 #Upper cutoff in model
    used_cache: bool = False
    ema_decay: float = 1.0
    precision: str = "32"
    unit: float = 1
    # nidek related
    activation: str = 'silu'
    remove_init: bool=False
    remove_atomref_energy:bool=False
    debug: bool = False
    test_energy_hami: bool = False
    test_homo_lumo_hami: bool = False
    num_sanity_val_steps: int = 0
