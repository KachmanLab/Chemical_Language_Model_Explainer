import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
from dagshub.pytorch_lightning import DAGsHubLogger
from src.dataloader import AqSolDataset
from src.model import AqueousRegModel, BaselineAqueousModel

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

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

dagslogger = DAGsHubLogger(
    name='results/aqueous-solu',
    metrics_path="/workspace/results/aqueous-solu/aqueous_metrics.csv",
    hparams_path="/workspace/results/aqueous-solu/aqueous_config.json",
    version='aqueous-v1'
    # default_save_path='/workspace/results/aqueous-solu',
)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='gpu',
    gpus=1,
    precision=16,
    logger=dagslogger,
    auto_lr_find=True,
)

trainer.fit(ft_model, train_loader, val_loader)
trainer.test(ft_model, test_loader)