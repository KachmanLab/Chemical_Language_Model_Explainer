import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.dataloader import AqSolDataset
from src.model import AqueousRegModel, BaselineAqueousModel

cfg = {
    'n_batch': 48,
    'seed': 42,
    'n_epochs': 50,
    'acc_test': True,
    'split': 0.9,
}

pl.seed_everything(cfg['seed'])

train_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train', cfg['acc_test'], 
    cfg['split'], data_seed=cfg['seed'], augment=False)
val_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'valid', cfg['acc_test'], 
    cfg['split'], data_seed=cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test', cfg['acc_test'], 
    cfg['split'], data_seed=cfg['seed'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'], 
    shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

ft_model = AqueousRegModel()
# ft_model = BaselineAqueousModel()
# unfreeze to train the whole model instead of just the head
ft_model.mmb.unfreeze() 

wandb_logger = WandbLogger(project='aqueous-solu', dir='/workspace/results/')
wandb_logger.experiment.config.update(cfg)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='gpu',
    gpus=1,
    precision=16,
    logger=wandb_logger,
    auto_lr_find=True,
)

trainer.fit(ft_model, train_loader, val_loader)
trainer.test(ft_model, test_loader)