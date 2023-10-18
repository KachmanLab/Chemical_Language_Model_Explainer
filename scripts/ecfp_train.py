import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import dagshub
# from dagshub.pytorch_lightning import DAGsHubLogger
import mlflow
from src.dataloader import AqSolECFP
from src.model import ECFPLinear

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

train_dataset = AqSolECFP('/workspace/data/AqueousSolu.csv', 'train', cfg['acc_test'], 
    cfg['split'], data_seed=cfg['seed'])
val_dataset = AqSolECFP('/workspace/data/AqueousSolu.csv', 'valid', cfg['acc_test'], 
    cfg['split'], data_seed=cfg['seed'])
test_dataset = AqSolECFP('/workspace/data/AqueousSolu.csv', 'test', cfg['acc_test'], 
    cfg['split'], data_seed=cfg['seed'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'], 
    shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

model = ECFPLinear(head=cfg['head'])

dagshub.init("Chemical_Language_Model_Explainer", "stefanhoedl", mlflow=True)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='cpu',
    # gpus=1,
    # precision=16,
    auto_lr_find=True,
)

with mlflow.start_run() as run:
    mlflow.pytorch.autolog(log_models=False)
    mlflow.log_params(cfg)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    mdir = 'ecfp'
    modelpath = f'/workspace/results/aqueous/models/ecfp_{run.info.run_id}.pt'

    trainer.save_checkpoint(modelpath)
 
print(model.head.fc1.weight)
