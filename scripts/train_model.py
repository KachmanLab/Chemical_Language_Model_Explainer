import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.model import AqueousRegModel, BaselineAqueousModel, ECFPLinear
import pickle
import dvc.api

root = "/workspace/results/aqueous/models"
# with open("/workspace/cfg/model_config.json", 'r') as f:
#     cfg = json.load(f)
cfg = dvc.api.params_show()
print(cfg)
print(cfg['ml']['model'], cfg['ml']['head'])
print(cfg['ds']['task'], cfg['ds']['split'])
pl.seed_everything(cfg['ml']['seed'])
root = f"/workspace/data/{cfg['ds']['task']}/{cfg['ds']['split']}"

with open(f"{root}/test.pkl", 'rb') as f:
    test = pickle.load(f)
test_loader = DataLoader(test, batch_size=cfg['n_batch'],
                         shuffle=False, num_workers=8)

for fold in range(cfg['n_splits']):
    # Load pickle
    with open(f"{root}/train{fold}.pkl", 'rb') as f:
        train = pickle.load(f)
    with open(f"{root}/valid{fold}.pkl", 'rb') as f:
        valid = pickle.load(f)
    train_loader = DataLoader(train, batch_size=cfg['n_batch'],
                              shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid, batch_size=cfg['n_batch'],
                              shuffle=False, num_workers=8)

    # configure regression head
    if 'hier' in cfg['head']:
        head = 'hier'
    elif 'lin' in cfg['head']:
        head = 'lin'
    print(cfg['head'], head)

    # configure model
    if cfg['model'] == 'mmb':
        model = AqueousRegModel(head=head)
    elif cfg['finetune'] or 'ft' in cfg['model'] or cfg['model'] == 'mmb-ft':
        # unfreeze to train the whole model instead of just the head
        # cfg['finetune'] = True
        model = AqueousRegModel(head=head)
        model.mmb.unfreeze()
    elif cfg['model'] == 'mmb-avg':
        model = BaselineAqueousModel(head=head)
    elif cfg['model'] == 'ecfp':
        # model = ECFPLinear(head=head)
        model = ECFPLinear(head=cfg['head'], dim=cfg['nbits'])
        # split train script into sklearn - vs - torch

    wandb_logger = WandbLogger(
        project='aqueous-solu' if cfg['property'] == 'aq' else cfg['property']
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

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    basepath = f"/workspace/{cfg['property']}_prod/{cfg['split']}"
    ft = '-ft' if (cfg['finetune'] and cfg['model'] == 'mmb') else ''
    mdir = f"{cfg['model']}{ft}-{head}"

    if cfg['finetune']:
        path = f"{basepath}/{mdir}/mmb{fold}.pt"
        trainer.save_checkpoint(path)
    else:
        path = f"{basepath}/{mdir}/head{fold}.pt"
        torch.save(model.head.state_dict(), path)

        artifact = wandb.Artifact(f"head{fold}_{mdir}", type='model')
        artifact.add_file(path)
        wandb_logger.experiment.log_artifact(artifact)

    # model.head.load_state_dict(torch.load(
    #                f"{basepath}/aq_head{i}_{suffix}.pt")
