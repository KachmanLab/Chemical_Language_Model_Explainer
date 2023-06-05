import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dagshub.pytorch_lightning import DAGsHubLogger
from src.dataloader import CombiSoluDataset
from src.model import CombiRegModel
import json

with open('/workspace/scripts/combi_config.json', 'r') as f:
    cfg = json.load(f)

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
# unfreeze to train the whole model instead of just the head
combi_model.mmb.unfreeze()

dagslogger = DAGsHubLogger(
    name='results/combi-solu',
    metrics_path="/workspace/results/combi-solu/combi_metrics.csv",
    hparams_path="/workspace/results/combi-solu/combi_config.json",
    version='combi-v1'
)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='gpu',
    gpus=1,
    precision=16,
    logger=dagslogger,
    auto_lr_find=True,
)

trainer.fit(combi_model, train_loader, val_loader)
trainer.test(combi_model, test_loader)