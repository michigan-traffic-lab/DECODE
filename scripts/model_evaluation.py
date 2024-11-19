import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from domain_expansion.model import build_model
from unitraj.datasets import build_dataset
from unitraj.utils.utils import set_seed
import hydra
from omegaconf import OmegaConf

# see the definition of the config.yaml file in the configs directory to modify the default values
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def eval(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['train_data_path'] = cfg['domain']['train_data_path']
    cfg['val_data_path'] = cfg['domain']['val_data_path']

    model = build_model(cfg).cuda()
    val_sets = []
    all_val_data_paths = cfg.val_data_path
    for val_data_path in all_val_data_paths:
        cfg['val_data_path'] = [val_data_path]
        val_set = build_dataset(cfg, val=True)
        val_sets.append(val_set)
    
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices) // val_sets[0].data_chunk_size, 1)
    val_loaders = [DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn) for val_set in val_sets]
    if cfg.ckpt_path is None:
        raise ValueError("Please provide a checkpoint path")
    checkpoint = torch.load(cfg.ckpt_path) # Replace with your checkpoint path
    model.load_state_dict(checkpoint['state_dict'])
    cfg.debug = True
    trainer = pl.Trainer(
        max_epochs=40,
        logger=None,
        profiler="simple",
        enable_model_summary=True,
        accelerator='gpu',
        num_nodes=1,
    )
    trainer.validate(model, dataloaders=val_loaders)


if __name__ == '__main__':
    eval()
