import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import dagshub
# from dagshub.pytorch_lightning import DAGsHubLogger
import mlflow
from src.dataloader import CombiSoluDataset
from src.model import CombiRegModel

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


dagshub.init("Chemical_Language_Model_Explainer", "stefanhoedl", mlflow=True)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='gpu',
    gpus=1,
    precision=16,
    auto_lr_find=True,
)

with mlflow.start_run() as run:
    mlflow.pytorch.autolog(log_models=False)
    mlflow.log_params(cfg)
    
    trainer.fit(combi_model, train_loader, val_loader)
    trainer.test(combi_model, test_loader)

    modelpath = f'/workspace/results/combi-solu/combi_{run.info.run_id}.pt'
    trainer.save_checkpoint(modelpath)
    mlflow.log_artifact(modelpath)