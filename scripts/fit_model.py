import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
import wandb
# from src.model import (
#     ECFPModel
# )
from src.dataloader import ECFPDataSplit
import pickle
import json
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def fit(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    print('FIT CONFIG from params.yaml')
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
        train = ECFPDataSplit(train, nbits=cfg.model.nbits)
        valid = ECFPDataSplit(valid, nbits=cfg.model.nbits)
        print('len train, val', len(train), len(valid))

        # configure model
        assert 'ecfp' in cfg.model.model and cfg.head.head in ['svr', 'rf']
        if cfg.head.head == 'svr':
            model = SVR(kernel='rbf')
        elif cfg.head.head == 'rf':
            model = RandomForestRegressor(n_estimators=100,
                                          random_state=42)

        model.fit(train.ecfp, train.labels)

        print('validating fold', fold)
        valid_preds = model.predict(valid.ecfp)
        val_rmse = mean_squared_error(valid_preds, valid.labels, squared=False)
        val_mse = mean_squared_error(valid_preds, valid.labels, squared=True)
        val_mae = mean_absolute_error(valid_preds, valid.labels)

        metrics[fold]['val_mae'] = val_mae
        metrics[fold]['val_mse'] = val_mse
        metrics[fold]['val_rmse'] = val_rmse

        path = f"{basepath}/{mdir}/model/head{fold}.pt"
        with open(path, 'wb') as file:
            pickle.dump(model, file)


    # select best model based on valid score, test of test set, save best ckpt
    best_fold = np.argmin([v['val_mae'] for k, v in metrics.items()])
    print(metrics)
    print('best fold was fold', best_fold)
    metrics['best_fold'] = str(best_fold)

    ckpt_path = f"{basepath}/{mdir}/model/head{best_fold}.pt"

    with open(ckpt_path, 'rb') as file:
        model = pickle.load(file)

    with open(f"{basepath}/{mdir}/best.pt", 'wb') as file:
        pickle.dump(model, file)

    test_preds = model.predict(test.ecfp)
    test_rmse = mean_squared_error(test_preds, test.labels, squared=False)
    test_mse = mean_squared_error(test_preds, test.labels, squared=True)
    test_mae = mean_absolute_error(test_preds, test.labels)

    metrics['test']['test_mae'] = test_mae
    metrics['test']['test_mse'] = test_mse
    metrics['test']['test_rmse'] = test_rmse
    print(metrics)
    with open(f"{basepath}/{mdir}/metrics.json", 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    fit()
