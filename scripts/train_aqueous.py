import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.dataloader import AqSolDataset
from src.model import AqueousRegModel, BaselineAqueousModel

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

train_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train',
    cfg['split_type'], cfg['split'], data_seed=cfg['seed'], augment=False)
val_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'valid',
    cfg['split_type'], cfg['split'], data_seed=cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test',
    cfg['split_type'], cfg['split'], data_seed=cfg['seed'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'],
                          shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'],
                        shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'],
                         shuffle=False, num_workers=8)

if cfg['model'] == 'mmb':
    if cfg['head'] == 'masked':
        head = 'hier_m'
    elif cfg['head'] == 'maskedlinear':
        head = 'linear'
    else:
        head = 'hier'
    model = AqueousRegModel(head=head)
    print(cfg['head'], head)
elif cfg['model'] == 'shap':
    model = BaselineAqueousModel()

if cfg['finetune']:
    # unfreeze to train the whole model instead of just the head
    model.mmb.unfreeze()

wandb_logger = WandbLogger(project='aqueous-solu',
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

mdir = 'aqueous' if cfg['model'] == 'mmb' else 'shap'

suffix = f"{cfg['model']}_{cfg['head']}_{cfg['split_type']}_{cfg['seed']}"
basepath = f"/workspace/results/aqueous/models"

if cfg['finetune']:
    modelpath = f"{basepath}/aqueous_{suffix}.pt"
    trainer.save_checkpoint(modelpath)
else:
    i=0
    headpath = f"{basepath}/aq_head{i}_{suffix}.pt"
    torch.save(model.head.state_dict(), headpath)

    artifact = wandb.Artifact(f"aq_head{i}", type='model')
    artifact.add_file(headpath)
    wandb_logger.experiment.log_artifact(artifact)

# model.head.load_state_dict(torch.load(
#                f"{basepath}/aq_head{i}_{suffix}.pt")
