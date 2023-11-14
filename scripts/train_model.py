import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.model import AqueousRegModel, BaselineAqueousModel, ECFPLinear
from src.dataloader import ECFPDataSplit
import pickle
import json
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(version_base="1.3",
            config_path="/workspace/conf",
            config_name="config")
def train(cfg: DictConfig) -> None:
    print('CONFIG')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    root = f"/workspace/data/{cfg.task.task}/{cfg.split.split}"

    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)
        if cfg.model.model == 'ecfp':
            test = ECFPDataSplit(test, nbits=cfg.model.nbits)
    test_loader = DataLoader(test, batch_size=cfg.model.n_batch,
                             shuffle=False, num_workers=8)

    metrics = {}
    for fold in range(cfg.split.n_splits):
        # Load pickle
        with open(f"{root}/train{fold}.pkl", 'rb') as f:
            train = pickle.load(f)
        with open(f"{root}/valid{fold}.pkl", 'rb') as f:
            valid = pickle.load(f)
        if cfg.model.model == 'ecfp':
            train = ECFPDataSplit(train, nbits=cfg.model.nbits)
            valid = ECFPDataSplit(valid, nbits=cfg.model.nbits)
        print('len train, val', len(train), len(valid))
        train_loader = DataLoader(train, batch_size=cfg.model.n_batch,
                                  shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid, batch_size=cfg.model.n_batch,
                                  shuffle=False, num_workers=8)

        # if fold > 0:
        #     del model
        #     del trainer
        #     del wandb_logger
        #     torch.cuda.empty_cache()
        #     gc.collect()

        # configure model
        if cfg.model.model == 'mmb':
            model = AqueousRegModel(head=cfg.head.head)
        elif cfg.model.finetune or cfg.model.model == 'mmb-ft':
            # unfreeze to train the whole model instead of just the head
            # cfg['finetune'] = True
            model = AqueousRegModel(head=cfg.head.head)
            model.mmb.unfreeze()
        elif cfg.model.model == 'mmb-avg':
            model = BaselineAqueousModel(head=cfg.head.head)
        elif cfg.model.model == 'ecfp':
            # model = ECFPLinear(head=head)
            model = ECFPLinear(head=cfg.head.head,
                               dim=cfg.model.nbits)
            # split train script into sklearn - vs - torch

        wandb_logger = WandbLogger(
            project='aqueous-solu' if cfg.task.task == 'aq' else cfg.task.task
        )
        wandb_logger.experiment.config.update(cfg)

        trainer = pl.Trainer(
            max_epochs=cfg.model.n_epochs,
            accelerator='gpu',
            gpus=1,
            precision=16,
            logger=wandb_logger,
            auto_lr_find=True,
        )

        trainer.fit(model, train_loader, valid_loader)
        metrics[fold] = trainer.validate(model, valid_loader)[0]

        basepath = f"/workspace/out/{cfg.task.task}/{cfg.split.split}"
        mdir = f"{cfg.model.model}-{cfg.head.head}"

        if cfg.model.finetune:
            path = f"{basepath}/{mdir}/model/mmb{fold}.pt"
            trainer.save_checkpoint(path)
        else:
            path = f"{basepath}/{mdir}/model/head{fold}.pt"
            torch.save(model.head.state_dict(), path)
            # artifact = wandb.Artifact(f"head{fold}_{mdir}", type='model')
            # artifact.add_file(path)
            # wandb_logger.experiment.log_artifact(artifact)

    # select best model based on valid score, test of test set, save best ckpt
    # TODO fix RMSE and change to val_rmse
    # best_fold = torch.argmax([v['val_mae'] for k, v in metrics.items()])

    print(metrics)
    best_fold = [v['val_mae'] for k, v in metrics.items()]
    print(best_fold)
    if len(best_fold) > 1:
        best_fold = np.argmax(best_fold)
    else:
        best_fold = 0
    ckpt_path = f"{basepath}/{mdir}/model/head{best_fold}.pt"

    if cfg.model.finetune or cfg.model.model == 'mmb-ft':
        model = model.load_from_checkpoint(ckpt_path, head=cfg.head.head)
        trainer.save_checkpoint(f"{basepath}/{mdir}/best.pt")
    else:
        model.head.load_state_dict(torch.load(ckpt_path))
        torch.save(model.head.state_dict(), f"{basepath}/{mdir}/best.pt")

    metrics['valid'] = trainer.validate(model, valid_loader)[0]
    metrics['test'] = trainer.test(model, test_loader)[0]
    print(metrics)
    with open(f"{basepath}/{mdir}/metrics.json", 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    train()
