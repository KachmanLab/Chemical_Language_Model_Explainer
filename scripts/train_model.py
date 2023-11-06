import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.model import AqueousRegModel, BaselineAqueousModel, ECFPLinear
import pickle
import dvc.api
import json

# @hydra.main(version_base=None, config_path='cfg', config_name='config')
# def main(cfg: DictConfig = None):
#     if cfg is None:
#     cfg = dvc.api.params_show()
#     return cfg
# cfg = get_config()

cfg = dvc.api.params_show()
print('ds', cfg['ml']['model'], cfg['ml']['head'])
print('ml', cfg['ds']['task'], cfg['ds']['split'])

pl.seed_everything(cfg['ml']['seed'])
root = f"/workspace/data/{cfg['ds']['task']}/{cfg['ds']['split']}"

with open(f"{root}/test.pkl", 'rb') as f:
    test = pickle.load(f)
test_loader = DataLoader(test, batch_size=cfg['ml']['n_batch'],
                         shuffle=False, num_workers=8)
metrics = {}

for fold in range(cfg['ds']['n_splits']):
    # Load pickle
    with open(f"{root}/train{fold}.pkl", 'rb') as f:
        train = pickle.load(f)
    with open(f"{root}/valid{fold}.pkl", 'rb') as f:
        valid = pickle.load(f)
    train_loader = DataLoader(train, batch_size=cfg['ml']['n_batch'],
                              shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid, batch_size=cfg['ml']['n_batch'],
                              shuffle=False, num_workers=8)

    # configure model
    if cfg['ml']['model'] == 'mmb':
        model = AqueousRegModel(head=cfg['ml']['head'])
    elif cfg['ml']['finetune'] or cfg['ml']['model'] == 'mmb-ft':
        # unfreeze to train the whole model instead of just the head
        # cfg['finetune'] = True
        model = AqueousRegModel(head=cfg['ml']['head'])
        model.mmb.unfreeze()
    elif cfg['ml']['model'] == 'mmb-avg':
        model = BaselineAqueousModel(head=cfg['ml']['head'])
    elif cfg['ml']['model'] == 'ecfp':
        # model = ECFPLinear(head=head)
        model = ECFPLinear(head=cfg['ml']['head'],
                           dim=cfg['ml']['nbits'])
        # split train script into sklearn - vs - torch

    wandb_logger = WandbLogger(
        project='aqueous-solu' if cfg['ds']['task'] == 'aq' else cfg['ds']['task']
    )
    wandb_logger.experiment.config.update(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg['ml']['n_epochs'],
        accelerator='gpu',
        gpus=1,
        precision=16,
        logger=wandb_logger,
        auto_lr_find=True,
    )

    trainer.fit(model, train_loader, valid_loader)
    metrics[fold] = trainer.validate(model, valid_loader)

    basepath = f"/workspace/out/{cfg['ds']['property']}/{cfg['ds']['split']}"
    mdir = f"{cfg['ml']['model']}-{cfg['ml']['head']}"

    if cfg['ml']['finetune']:
        path = f"{basepath}/{mdir}/mmb{fold}.pt"
        trainer.save_checkpoint(path)
    else:
        path = f"{basepath}/{mdir}/head{fold}.pt"
        torch.save(model.head.state_dict(), path)

        artifact = wandb.Artifact(f"head{fold}_{mdir}", type='model')
        artifact.add_file(path)
        wandb_logger.experiment.log_artifact(artifact)

# select best model based on valid score, test of test set, save best ckpt
# TODO fix RMSE and change to val_rmse
best_fold = torch.argmax([metrics.get(f)['val_mae'] for f in range(
    cfg['ds']['n_splits'])])
ckpt_path = f"{basepath}/{mdir}/head{best_fold}.pt"

if cfg['ml']['model'] == 'mmb-ft' or cfg['ml']['finetune']:
    model = model.load_from_checkpoint(ckpt_path, head=cfg['ml']['head'])
else:
    model.head.load_state_dict(torch.load(ckpt_path))

metrics['valid'] = trainer.validate(model, valid_loader)
metrics['test'] = trainer.test(model, test_loader)
print(metrics)
with open(f"{basepath}/{mdir}/metrics.json", 'w') as f:
    json.dump(metrics, f)
