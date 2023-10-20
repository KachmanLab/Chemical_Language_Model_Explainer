import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
#import dagshub
# from dagshub.pytorch_lightning import DAGsHubLogger
#import mlflow
from pytorch_lightning.loggers import WandbLogger
from src.dataloader import AqSolECFP
from src.model import ECFPLinear

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

train_dataset = AqSolECFP('/workspace/data/AqueousSolu.csv', 'train',
    cfg['acc_test'], cfg['split'], cfg['seed'], cfg['nbits'])
val_dataset = AqSolECFP('/workspace/data/AqueousSolu.csv', 'valid',
    cfg['acc_test'], cfg['split'], cfg['seed'], cfg['nbits'])
test_dataset = AqSolECFP('/workspace/data/AqueousSolu.csv', 'test',
    cfg['acc_test'], cfg['split'], cfg['seed'], cfg['nbits'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'], 
    shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

model = ECFPLinear(head=cfg['head'], dim=cfg['nbits'])

wandb_logger = WandbLogger(project='aqueous-solu',
                           #dir='/results/aqueous/models/'
                           )
wandb_logger.experiment.config.update(cfg)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='cpu',
    # gpus=1,
    # precision=16,
    logger=wandb_logger,
    auto_lr_find=True,
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

mdir = 'ecfp'
savedir = "/workspace/results/aqueous/models"
head = 'linear' if cfg['head']=='linear' else ''
modelpath = f"{savedir}/ecfp_{cfg['nbits']}{head}.pt"
trainer.save_checkpoint(modelpath)

print(model.head.fc1.weight)
