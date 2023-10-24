import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import dagshub
# from dagshub.pytorch_lightning import DAGsHubLogger
import mlflow
from src.prop_loader import CMCDataset
from src.model import AqueousRegModel, BaselineAqueousModel

with open('/workspace/scripts/cmc_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

train_dataset = CMCDataset('/workspace/data/cmcdata.csv', 'train',
                    cfg['split_type'], cfg['split'], data_seed=cfg['seed'], augment=False)
val_dataset = CMCDataset('/workspace/data/cmcdata.csv', 'valid',
                    cfg['split_type'], cfg['split'], data_seed=cfg['seed'])
test_dataset = CMCDataset('/workspace/data/cmcdata.csv', 'test',
                    cfg['split_type'], cfg['split'], data_seed=cfg['seed'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'],
                          shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'],
                        shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'],
                         shuffle=False, num_workers=8)

if cfg['model'] == 'mmb':
    model = AqueousRegModel(head=cfg['head'])
elif cfg['model'] == 'shap':
    model = BaselineAqueousModel()

# optionally unfreeze to train the whole model instead of just the head
if cfg['finetune']:
    # unfreeze to train the whole model instead of just the head
    model.mmb.unfreeze()

wandb_logger = WandbLogger(project='cmc',
                           #dir='/results/aqueous/models/'
                           )
wandb_logger.experiment.config.update(cfg)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='gpu',
    gpus=1,
    precision=16,
    logger=wandb_logger,
    auto_lr_find=True,
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

suffix = f"{cfg['model']}_{cfg['head']}_{cfg['split_type']}"
modelpath = f"/workspace/results/cmc/models/cmc_{suffix}.pt"
trainer.save_checkpoint(modelpath)
    # mlflow.log_artifact(modelpath)
