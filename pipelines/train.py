import numpy as np  # sometimes needed to avoid mkl-service error
import sys
import os
import glob
import pytorch_lightning as pl
import glob
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from swanlab.integration.pytorch_lightning import SwanLabLogger as WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path

import torch
import swanlab as wandb
import os
# os.environ["WANDB_MODE"] = "offline"
# print(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH')}")  

from datetime import datetime
import random
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src in root folder
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))

from src.utility.hydra_config import Config
from src.training.module import LNNP
from src.training.logger import CSVLogger, get_latest_ckpt
from src.utility.callbacks import EMA
from src.training.data import DataModule

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.overrides")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._check")

# import shutup
# shutup.please()

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def cli(config: DictConfig) -> None:
    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, config)
    OmegaConf.set_struct(config, False)

    if config.job_id == "auto":
        ct = datetime.now()
        #generate a random word like flushing river
        config.job_id = f'time{ct.year}_{ct.month}_{ct.day}_{ct.hour}_{ct.minute}_{ct.second}'


    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)
    
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path, exist_ok=True)

    with open(config.log_dir+"/config.yaml", 'w') as file:OmegaConf.save(config=config, f=file.name)


    config.batch_size = config.batch_size // config.ngpus if config.ngpus > 1 else config.batch_size
    if config.inference_batch_size is None:
        config.inference_batch_size = config.batch_size

    # save_argparse(config, os.path.join(config.log_dir, "input.yaml"),[])
    main(config)


def main(config):
    
    pl.seed_everything(config.seed, workers=True)
    # initialize data module
    data = DataModule(config)

    # initialize lightning module
    # create of SPHNet model
    model = LNNP(config)

    callbacks = []
    callbacks.append(EarlyStopping("val_loss", patience=config.early_stopping_patience,
                                   check_on_train_epoch_end=False))
    # Rule 1: save first 10 epochs unconditionally
    class _First10Callback(pl.Callback):
        def on_validation_end(self, trainer, pl_module):
            if trainer.current_epoch < 10:
                trainer.save_checkpoint(
                    os.path.join(trainer.default_root_dir,
                                 f"first10-epoch{trainer.current_epoch:02d}-{trainer.callback_metrics.get('val_loss', 0):.6f}.ckpt"))
    callbacks.append(_First10Callback())
    # Rule 2: save every 20 epochs, keep all
    callbacks.append(ModelCheckpoint(
        dirpath=config.log_dir,
        monitor="val_loss",
        save_top_k=-1,
        every_n_epochs=20,
        mode="min",
        filename="{step}-epoch{epoch:04d}-{val_loss:.6f}",
    ))
    # Rule 3: save best + last
    callbacks.append(ModelCheckpoint(
        dirpath=config.log_dir,
        monitor="val_loss",
        save_top_k=1,
        every_n_epochs=1,
        mode="min",
        save_last=True,
        filename="best-{epoch:04d}-{val_loss:.6f}",
    ))
    latest_file = get_latest_ckpt(config.log_dir)
    print("latest_file is: ", latest_file)
    
    if config.ema_decay!=1:
        callbacks.append(EMA(decay=config.ema_decay))
        
        
    # logger    
    tb_logger = pl.loggers.TensorBoardLogger(
        config.log_dir, name="tensorbord", version="", default_hp_metric=False
    )
    csv_logger = CSVLogger(config.log_dir, name="", version="")
    # wandb is project/group/name format to save all the log
    wandb_logger = WandbLogger(
                               entity=None,
                               project=config.wandb.wandb_project,
                               group = config.wandb.wandb_group,
                               name=config.job_id, 
                               settings=wandb.Settings(start_method='fork', code_dir="."),
                               )

    # login into wandb
    @rank_zero_only
    def log_code():
        if config.wandb.open:
            wandb.login(api_key=config.wandb.wandb_api_key, save=True)
            run = wandb.init(
                project=config.wandb.wandb_project,
                experiment_name=config.wandb.wandb_name,
                config=config,
                logdir=config.log_dir,
            )
    log_code()

    if config.precision == '32':
        #ENABLE TENSOR CORES
       torch.set_float32_matmul_precision('highest') # set from highest to high # no we need highest

    strategy=DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        max_steps=config.max_steps,
        devices=list(range(config.ngpus)),
        num_nodes=config.num_nodes,
        default_root_dir=config.log_dir,
        callbacks=callbacks,
        logger=[tb_logger, wandb_logger,csv_logger], 
        val_check_interval = config.val_check_interval,
        check_val_every_n_epoch = config.check_val_every_n_epoch,
        precision=config.precision,
        strategy=strategy,
        gradient_clip_val = config.gradient_clip_val,
        use_distributed_sampler = False, # Manual sharding done inside datamodule.
        num_sanity_val_steps = config.num_sanity_val_steps,
        log_every_n_steps = config.get("log_every_n_steps", 50),
    )

    # use previous ckpt if have one
    ckpt_files = glob.glob(os.path.join(config.log_dir, '*.ckpt'))  
    print(os.path.join(config.log_dir, '*.ckpt'))
    if ckpt_files:  
        latest_file = max(ckpt_files, key=os.path.getctime)  
        if not config.finetune_flag:
            # when finetuning, the model initialization is moved to LNNP.__init__()
            # so the ckpt of trainer needs to be reset
            print(f"The latest .ckpt file is: {latest_file}")  
    else: 
        if not config.finetune_flag: 
            print("No .ckpt files found in the folder.")
        latest_file = None
    trainer.fit(model, data, ckpt_path=latest_file)

    # run test set after completing the fit
    if not config.get("skip_test", False):
        latest_file = get_latest_ckpt(config.log_dir)
        print(latest_file,config.log_dir)
        trainer.test(model, data,ckpt_path=latest_file)


if __name__ == "__main__":
     cli()
