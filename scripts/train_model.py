import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.model import (
    AqueousRegModel, BaselineAqueousModel, ECFPLinear, ECFPModel
)
from src.dataloader import ECFPDataSplit
import pickle
import json
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    print('TRAIN CONFIG from params.yaml')
    cfg = OmegaConf.load('./params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    root = f"./data/{cfg.task.task}/{cfg.split.split}"

    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)
        if cfg.model.model == 'ecfp':
            test = ECFPDataSplit(test, nbits=cfg.model.nbits)
    test_loader = DataLoader(test, batch_size=cfg.model.n_batch,
                             shuffle=False, num_workers=8)

    basepath = f"./out/{cfg.task.task}/{cfg.split.split}"
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    metrics = {}
    for fold in range(cfg.split.n_splits):
        # Load pickle
        with open(f"{root}/train{fold}.pkl", 'rb') as f:
            train = pickle.load(f)
        with open(f"{root}/valid{fold}.pkl", 'rb') as f:
            valid = pickle.load(f)
        if 'ecfp' in cfg.model.model:
            train = ECFPDataSplit(train, nbits=cfg.model.nbits)
            valid = ECFPDataSplit(valid, nbits=cfg.model.nbits)
        print('len train, val', len(train), len(valid))
        train_loader = DataLoader(train, batch_size=cfg.model.n_batch,
                                  shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid, batch_size=cfg.model.n_batch,
                                  shuffle=False, num_workers=8)

        # configure model
        if 'ecfp' in cfg.model.model and cfg.head.head in ['svr', 'rf']:
            model = ECFPModel(head=cfg.head.head,
                              dim=cfg.model.nbits)
            model.fit(train)

        elif fold == 0:
            if cfg.model.model in ['mmb', 'mmb-ft']:
                model = AqueousRegModel(head=cfg.head.head,
                                        finetune=cfg.model.finetune)
                # _sanity_mmb = np.array(model.mmb.state_dict())
            elif cfg.model.model in ['mmb-avg', 'mmb-ft-avg']:
                model = BaselineAqueousModel(head=cfg.head.head,
                                             finetune=cfg.model.finetune)
            elif cfg.model.model == 'ecfp':
                model = ECFPLinear(head=cfg.head.head,
                                   dim=cfg.model.nbits)
                if cfg.head.head in ['lin', 'hier']:
                    model = ECFPLinear(head=cfg.head.head,
                                       dim=cfg.model.nbits)
            if cfg.model.finetune or 'ft' in cfg.model.model:
                # unfreeze to train the whole model instead of just the head
                # cfg['finetune'] = True
                torch.save(model.mmb.state_dict(),
                           f"{basepath}/{mdir}/model/mmb.pt")
        else:
            # only reset head instead of re-initializing full mmb model
            model.reset_head()
            if 'mmb' in cfg.model.model:
                model.mmb.freeze()
            if cfg.model.finetune or 'ft' in cfg.model.model:
                # restore base MMB core
                model.mmb.load_state_dict(
                    torch.load(f"{basepath}/{mdir}/model/mmb.pt"))
                model.mmb.unfreeze()

        wandb_logger = WandbLogger(
            project='aqueous-solu' if cfg.task.task == 'aq' else cfg.task.task
        )
        wandb_logger.experiment.config.update(cfg)
        wandb_logger.experiment.config.update({
            'split': cfg.split.split,
            'model': cfg.model.model,
            'head': cfg.head.head,
            'n_epochs': cfg.model.n_epochs,
        })
        wandb.watch(model, log_freq=16, log='all')

        trainer = pl.Trainer(
            max_epochs=cfg.model.n_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            gpus=1 if torch.cuda.is_available() else 0,
            precision=16 if torch.cuda.is_available() else 32,
            logger=wandb_logger,
            auto_lr_find=False,
        )

        if 'mmb' in cfg.model.model:
            trainer.fit(model, train_loader, valid_loader)
        # elif 'ecfp' in cfg.model.model and cfg.head.head in ['svr', 'rf']:
            # TODO merge with val set
            # feat, lab = train[:]
            # model.head.fit(feat, lab)

            # lasso = Lasso(alpha=0.2, fit_intercept=False)
        #     feat, lab = train[:]
        #     lasso.fit(feat, lab)
        #
        #     model.head.fc1.weight.requires_grad = False
        #     model.head.fc1.weight[0] = torch.FloatTensor(lasso.coef_)

        # trainer.fit(model, train_loader, valid_loader)

        print('validating fold', fold)
        metrics[fold] = trainer.validate(model, valid_loader)[0]
        # wandb.finish()

        path = f"{basepath}/{mdir}/model/head{fold}.pt"
        if cfg.model.finetune:
            # trainer.save_checkpoint(path)
            mmbpath = f"{basepath}/{mdir}/model/mmb{fold}.pt"
            torch.save(model.mmb.state_dict(), mmbpath)
            torch.save(model.head.state_dict(), path)
        else:
            torch.save(model.head.state_dict(), path)

    # select best model based on valid score, test of test set, save best ckpt
    # TODO fix RMSE and change to val_rmse
    best_fold = np.argmin([v['val_mae'] for k, v in metrics.items()])
    print(metrics)
    print('best fold was fold', best_fold)
    metrics['best_fold'] = str(best_fold)

    # with open(f"{basepath}/{mdir}/best.pt", 'wb') as f:
    #     tmp = torch.load(f"{basepath}/{mdir}/model/model{best_fold}.pt")
    #     torch.save(tmp, f)

    ckpt_path = f"{basepath}/{mdir}/model/head{best_fold}.pt"
    if cfg.model.finetune or cfg.model.model == 'mmb-ft':
        mmb_path = f"{basepath}/{mdir}/model/mmb{best_fold}.pt"
        model.mmb.load_state_dict(torch.load(mmb_path))
        model.head.load_state_dict(torch.load(ckpt_path))
        torch.save(model.mmb.state_dict(), f"{basepath}/{mdir}/best_mmb.pt")
        torch.save(model.head.state_dict(), f"{basepath}/{mdir}/best.pt")
    else:
        model.head.load_state_dict(torch.load(ckpt_path))
        # check mmb backbone is unchanged
        # assert np.allclose(np.array(model.mmb.state_dict()), _sanity_mmb)
        torch.save(model.head.state_dict(), f"{basepath}/{mdir}/best.pt")

    metrics['valid'] = trainer.validate(model, valid_loader)[0]
    metrics['test'] = trainer.test(model, test_loader)[0]
    print(metrics)
    with open(f"{basepath}/{mdir}/metrics.json", 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    train()
