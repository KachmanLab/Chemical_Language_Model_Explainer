import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.dataloader import CombiSoluDataset
from src.model import CombiRegModel

cfg = {
    'n_batch': 56,
    'seed': 42,
    'n_epochs': 40,
    'temp_test': False,
    'split': 0.9,
    'scale_logS': True,
}

pl.seed_everything(cfg['seed'])

train_dataset = CombiSoluDataset('/workspace/data/CombiSolu-Exp.csv', 'train', 
    cfg['temp_test'], cfg['split'], data_seed=cfg['seed'], scale_logS=cfg['scale_logS'])
val_dataset = CombiSoluDataset('/workspace/data/CombiSolu-Exp.csv', 'valid', 
    cfg['temp_test'], cfg['split'], data_seed=cfg['seed'], scale_logS=cfg['scale_logS'])
test_dataset = CombiSoluDataset('/workspace/data/CombiSolu-Exp.csv', 'test', 
    cfg['temp_test'], cfg['split'], data_seed=cfg['seed'], scale_logS=cfg['scale_logS'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'], 
    shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

combi_model = CombiRegModel()
combi_model.mmb.unfreeze()

wandb_logger = WandbLogger(project='combi-solu', dir='/workspace/results/')
wandb_logger.experiment.config.update(cfg)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='gpu',
    gpus=1,
    precision=16,
    logger=wandb_logger,
    auto_lr_find=True,
)

trainer.fit(combi_model, train_loader, val_loader)
trainer.test(combi_model, test_loader)