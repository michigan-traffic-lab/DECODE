import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from domain_expansion.model import build_model
from unitraj.datasets import build_dataset
from unitraj.utils.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os
from pathlib import Path
from datetime import datetime, timedelta

# see the definition of the config.yaml file in the configs directory to modify the default values
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['train_data_path'] = cfg['domain']['train_data_path']
    cfg['val_data_path'] = cfg['domain']['val_data_path']
    cfg['pretrained_model_path'] = cfg['method']['pretrained_model_path']
    cfg['output_path'] = cfg['method']['output_path']

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_sets = []
    all_val_data_paths = cfg.val_data_path
    for val_data_path in all_val_data_paths:
        cfg['val_data_path'] = [val_data_path]
        val_set = build_dataset(cfg, val=True)
        val_sets.append(val_set)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices) // train_set.data_chunk_size, 1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_sets[0].data_chunk_size, 1)

    call_backs = []

    output_root = Path(cfg['output_path']) / cfg['exp_name']
    version_name = (cfg['version_name'] + '_') if 'version_name' in cfg else ''
    version_name += datetime.now().strftime('%y%m%d_%H%M')
    ckpt_path = output_root / version_name
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=2,
        save_last=True,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loaders = [DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn) for val_set in val_sets]

    if cfg.method.strategy_name == 'ewc':
        model.train_loader = train_loader
        
    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else WandbLogger(project="DECODE", name=cfg.exp_name, id=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        accelerator= "gpu",
        profiler="simple",
        strategy="auto", #if cfg.debug else "ddp",
        callbacks=call_backs
    )
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loaders, ckpt_path=cfg.ckpt_path)



if __name__ == '__main__':
    train()
