# import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# import wandb
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
from sverad.sverad_svm import ExplainingSVR
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix

def evaluate(preds, labels, prefix=''):
    metric = {}
    mae = mean_absolute_error(preds, labels)
    mse = mean_squared_error(preds, labels, squared=True)
    rmse = mean_squared_error(preds, labels, squared=False)

    metric[f'{prefix}_mae'] = mae
    metric[f'{prefix}_mse'] = mse
    metric[f'{prefix}_rmse'] = rmse
    return metric


@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def train_sklearn(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    print('FIT SKLEARN CONFIG from params.yaml')
    cfg = OmegaConf.load('./params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    root = f"./data/{cfg.task.task}/{cfg.split.split}"

    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)
        if 'ecfp' in cfg.model.model:
            test = ECFPDataSplit(test, nbits=cfg.model.nbits)
            print(test.ecfp.shape, test.labels.shape)

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
        assert 'ecfp' in cfg.model.model and cfg.head.head in ['svr', 'rf', 'sverad']
        if cfg.head.head == 'svr':
            model = SVR(kernel='rbf',
                        # gamma=0.01,
                        # C=10
                        )
        elif cfg.head.head == 'sverad':
            model = ExplainingSVR(C=1.0)
            train.ecfp = csr_matrix(train.ecfp)
            valid.ecfp = csr_matrix(valid.ecfp)
            test.ecfp = csr_matrix(test.ecfp)
        elif cfg.head.head == 'rf':
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_split=2,
                                          min_samples_leaf=1,
                                          random_state=42)

        print('ecfptrain', train.ecfp.shape)
        print('lab', train.labels.shape)
        model.fit(train.ecfp, train.labels)

        # param_grid = {
        #     'C': [1, 10, 100, 1000],
        #     'gamma': [0.01, 0.1, 1, 'scale'],  # 'scale' is a good default
        #     'epsilon': [0.01, 0.1, 0.5]
        # }
        # grid_search = GridSearchCV(model, param_grid, cv=5)
        # grid_search.fit(train.ecfp, train.labels)
        # print("Best parameters:", grid_search.best_params_)

        print('validating fold', fold)
        valid_preds = model.predict(valid.ecfp)
        metrics[fold] = evaluate(valid_preds, valid.labels, 'val')

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
    metrics['test'] = evaluate(test_preds, test.labels, 'test')
    print(metrics)
    with open(f"{basepath}/{mdir}/metrics.json", 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    train_sklearn()
