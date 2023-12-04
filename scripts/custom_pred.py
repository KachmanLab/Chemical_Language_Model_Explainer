import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from src.model import AqueousRegModel, BaselineAqueousModel, ECFPLinear
from src.dataloader import DataSplit
import pickle
import hydra
import json
from omegaconf import OmegaConf, DictConfig
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns


def predict_custom():
    cfg = OmegaConf.load('./params.yaml')
    print('PREDICT CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))
    cfg.task.task = 'aq'

    pl.seed_everything(cfg.model.seed)
    root = f"./data/{cfg.task.task}/{cfg.split.split}"
    basepath = f"./out/{cfg.task.task}/{cfg.split.split}"
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    ckpt_path = f"{basepath}/{mdir}/best.pt"

    fileroot = "./data/organic_peroxides_smiles.csv"
    # test_ds = AqSolDataset('test',
    #     fileroot, 'SMILES', 'PRED',
    #     split='random', split_frac=0.01, n_splits=1,
    #     data_seed=42)
    # test_ds = test_ds[:]
    
    
    df = pd.read_csv(fileroot)
    test_ds = DataSplit(smiles=df['SMILES'],
                        labels=df['PRED'],
                        subset='perox')
    test_loader = DataLoader(test_ds, batch_size=cfg.model.n_batch,
                             shuffle=False, num_workers=8)

    head = cfg.head.head
    if cfg.model.model == 'mmb':
        model = AqueousRegModel(head=head,
                                finetune=cfg.model.finetune)
        model.head.load_state_dict(torch.load(ckpt_path))
    if cfg.model.finetune or cfg.model.model == 'mmb-ft':
        model = AqueousRegModel(head=head,
                                finetune=cfg.model.finetune)
        # model = model.load_from_checkpoint(ckpt_path, head=head)
        mmb_path = f"{basepath}/{mdir}/best_mmb.pt"
        model.mmb.load_state_dict(torch.load(mmb_path))
        model.head.load_state_dict(torch.load(ckpt_path))
    elif cfg.model.model == 'mmb-avg':
        model = BaselineAqueousModel(head=head,
                                     finetune=cfg.model.finetune)

    trainer = pl.Trainer(
        accelerator='gpu',
        gpus=1,
        precision=16,
    )

    # metrics[f'test_{best_fold}'] = trainer.validate(model, test_loader)[0]

    all = trainer.predict(model, test_loader)

    # results = pd.DataFrame(columns=[
    #     'SMILES', 'Tokens', 'Prediction', 'Label', 'Split']
    # )
    smiles = list(chain(*[f.get('smiles') for f in all]))
    tokens = list(chain(*[f.get('tokens') for f in all]))
    preds = torch.concat([f.get('preds') for f in all]).cpu().numpy()
    labels = torch.concat([f.get('labels') for f in all]).cpu().numpy()

    results = pd.DataFrame({
        'SMILES': smiles,
        'Prediction': preds,
        'Label': labels,
        'Split': 'perox'
        # 'Tokens': tokens,
        })
    # results = pd.concat([results, res], axis=0)

    # reset index to correspond to visualization UID
    results = results.reset_index(drop=True)
    results = results.reset_index().rename(columns={'index': 'uid'})
    results.to_csv(f"{basepath}/{mdir}/peroxide_preds.csv", index=False)

if __name__ == "__main__":
    predict_custom()
