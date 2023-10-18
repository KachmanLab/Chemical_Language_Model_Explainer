import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from rdkit.Chem import Draw, AllChem
# from rdkit import Chem, rdBase

import numpy as np
import pandas as pd
import os
import glob
import json
from itertools import chain

from src.dataloader import AqSolDataset
from src.model import AqueousRegModel

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])
train_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'], augment=False)
val_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'valid', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'], 
    shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

subfolders = [f.path for f in os.scandir('/workspace/results/aqueous/models/') \
    if (f.path.endswith('.pt') or f.path.endswith('.ckpt'))]
ckpt_path = max(subfolders, key=os.path.getmtime)

ft_model = AqueousRegModel()
ft_model = ft_model.load_from_checkpoint(ckpt_path)
ft_model.unfreeze()

trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    precision=16,
)

train = trainer.predict(ft_model, train_loader)
val = trainer.predict(ft_model, val_loader)
test = trainer.predict(ft_model, test_loader)

results = pd.DataFrame(
    columns=['SMILES', 'Tokens', 'logS_pred', 'logS_exp', 'Atom_weights', 'Split']
)
for split, all in list(zip(['test', 'val', 'train'], [test, val, train])):
    # reverse order for consistency with plotting
    smiles = list(chain(*[f.get('smiles') for f in all]))
    tokens = list(chain(*[f.get('tokens') for f in all]))
    atom_weights = list(chain(*[f.get('atom_weights') for f in all]))
    preds = torch.concat([f.get('preds') for f in all]).cpu().numpy()
    labels = torch.concat([f.get('labels') for f in all]).cpu().numpy()

    res = pd.DataFrame({
        'SMILES': smiles,
        'Tokens': tokens,
        'logS_pred': preds,
        'logS_exp': labels,
        'Atom_weights': atom_weights,
        'Split': split
        })
    results = pd.concat([results, res], axis=0)

# reset index to correspond to visualization UID
results = results.reset_index(drop=True)
results = results.reset_index().rename(columns={'index':'uid'})
results.to_csv('/workspace/results/aqueous/AqueousSolu_predictions.csv', index=False)
