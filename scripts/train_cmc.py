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
                           cfg['split'], data_seed=cfg['seed'], augment=False)
val_dataset = CMCDataset('/workspace/data/cmcdata.csv', 'valid',
                         cfg['split'], data_seed=cfg['seed'])
test_dataset = CMCDataset('/workspace/data/cmcdata.csv', 'test',
                          cfg['split'], data_seed=cfg['seed'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'],
                          shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'],
                        shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'],
                         shuffle=False, num_workers=8)

if cfg['model'] == 'cmc':
    ft_model = AqueousRegModel()
elif cfg['model'] == 'shap':
    ft_model = BaselineAqueousModel()
# optionally unfreeze to train the whole model instead of just the head
# ft_model.mmb.unfreeze()

# mlflow.set_tracking_uri("http://localhost:5555")
dagshub.init("Chemical_Language_Model_Explainer", "stefanhoedl", mlflow=False)

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

    trainer.fit(ft_model, train_loader, val_loader)
    trainer.test(ft_model, test_loader)

    mdir = 'cmc' if cfg['model'] == 'cmc' else 'shap'
    modelpath = f'/workspace/results/cmc/models/{mdir}_{run.info.run_id}.pt'
    trainer.save_checkpoint(modelpath)
    # mlflow.log_artifact(modelpath)
