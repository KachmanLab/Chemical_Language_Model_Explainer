import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.model import AqueousRegModel, BaselineAqueousModel, ECFPLinear
import pickle
import dvc.api

# import json
# with open("/workspace/cfg/model_config.json", 'r') as f:
#     cfg = json.load(f)

cfg = dvc.api.params_show()
print(cfg)
print(cfg['ml']['model'], cfg['ml']['head'])
print(cfg['ml']['model'], cfg['ml']['head'])
print(cfg['ds']['task'], cfg['ds']['split'])

pl.seed_everything(cfg['ml']['seed'])
root = f"/workspace/data/{cfg['ds']['task']}/{cfg['ds']['split']}"

with open(f"{root}/test.pkl", 'rb') as f:
    test = pickle.load(f)
test_loader = DataLoader(test, batch_size=cfg['ml']['n_batch'],
                         shuffle=False, num_workers=8)

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

    # configure regression head
    # if 'hier' in cfg['ml']['head']:
    #     head = 'hier'
    # elif 'lin' in cfg['ml']['head']:
    #     head = 'lin'
    # print(cfg['ml']['head'], head)

    # configure model
    if cfg['ml']['model'] == 'mmb':
        model = AqueousRegModel(head=cfg['ml']['head'])
    elif cfg['ml']['finetune'] or 'ft' in cfg['ml']['model'] or cfg['ml']['model'] == 'mmb-ft':
        # unfreeze to train the whole model instead of just the head
        # cfg['finetune'] = True
        model = AqueousRegModel(head=cfg['ml']['head'])
        model.mmb.unfreeze()
    elif cfg['ml']['model'] == 'mmb-avg':
        model = BaselineAqueousModel(head=cfg['ml']['head'])
    elif cfg['ml']['model'] == 'ecfp':
        # model = ECFPLinear(head=head)
        model = ECFPLinear(head=cfg['ml']['head'], dim=cfg['ml']['nbits'])
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
    trainer.test(model, test_loader)

    basepath = f"/workspace/{cfg['ds']['property']}_prod/{cfg['ds']['split']}"
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

    # model.head.load_state_dict(torch.load(
    #                f"{basepath}/aq_head{i}_{suffix}.pt")
