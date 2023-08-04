import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from rdkit.Chem import Draw, AllChem
from rdkit import Chem, rdBase

import seaborn as sns
import numpy as np
import pandas as pd
import os
import glob
import json
from typing import List

from src.dataloader import AqSolDataset
from src.model import BaselineAqueousModel
from src.explainer import ColorMapper, plot_weighted_molecule 
from nemo_src.regex_tokenizer import RegExTokenizer
import shap

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)
cfg['n_batch'] = 4

pl.seed_everything(cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)
    
subfolders = [f.path for f in os.scandir('/workspace/results/shap/models/') \
    if (f.path.endswith('.pt') or f.path.endswith('.ckpt'))]
ckpt_path = max(subfolders, key=os.path.getmtime)

ft_model = BaselineAqueousModel().load_from_checkpoint(ckpt_path)
ft_model.unfreeze()
mmb_tokenizer = ft_model.tokenizer

class ShapTokenizer(RegExTokenizer):
    """This minimal subset means the tokenizer must return a 
    dictionary with ‘input_ids’ and then either include an 
    ‘offset_mapping’ entry in the same dictionary or provide 
    a .convert_ids_to_tokens or .decode method."""

    def __init__(self):
        super().__init__()
        self.load_tokenizer()

    def convert_ids_to_tokens(self, ids):
        return self.ids_to_tokens(ids)

    def tokenize_one(self, smi: str):
        token = self.text_to_tokens(smi) 
        token_id = self.token_to_ids(token)
        # print('**', token_id)

        pad_length = 0
        encoder_mask = (
            [1] * len(token_id)) + ([0] * (pad_length - len(token_id))
        )
        token_id = torch.tensor(token_id, dtype=torch.int64).cuda()
        encoder_mask = torch.tensor(encoder_mask,
                                    dtype=torch.int64,
                                    device=token_id.device)

        return token_id, encoder_mask

    def __call__(self, text):
        token_ids, token_masks = self.tokenize_one(text)
        return {'input_ids': token_ids.tolist(),
                'input_masks': token_masks}

tokenizer = ShapTokenizer()
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(
    ft_model,
    masker
)

test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

results = pd.DataFrame(
    columns=['SMILES', 'Tokens', 'logS_pred', 'logS_exp', 'SHAP_weights', 'Split']
)
cmapper = ColorMapper()

for batch in test_loader:
    smiles, labels = batch
    print(smiles)
    shapvals = explainer(smiles).values
    tokens = [tokenizer.text_to_tokens(s) for s in smiles]
    preds = ft_model(smiles).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    # print('**', smiles, labels, tokens, preds, shapvals)
    # assert all([len(t) == len(s) for t, s in zip(tokens, shapvals)])

    ###############################
    # plot all molecules in batch #
    for b_ix in range(len(smiles)):
        token = tokens[b_ix]
        smi = smiles[b_ix]
        lab = labels[b_ix]
        pred = preds[b_ix]
        uid = len(results) + b_ix

        atom_color = cmapper(shapvals[b_ix], tokens[b_ix])
        atom_color = cmapper.to_rdkit_cmap(atom_color)
        
        if uid not in [39, 94, 210, 217]:
            # segmentation fault, likely due to weird structure?
            plot_weighted_molecule(atom_color, smi, token, lab, pred, 
            f"{uid}_shap", f'/workspace/results/shap/viz')
    ###############################
    
    res = pd.DataFrame({
        'SMILES': smiles,
        'Tokens': tokens,
        'logS_pred': preds,
        'logS_exp': labels,
        'SHAP_weights': shapvals,
        'Split': 'test'
        })
    print(res)
    results = pd.concat([results, res], axis=0)

print('token/shap equal len', all([len(t) == len(s) for t, s in zip(
    results.Tokens, results.SHAP_weights
)]))

results = results.reset_index(drop=True)
results = results.reset_index().rename(columns={'index':'uid'})
results.to_csv('/workspace/results/shap/AqueousSolu_SHAP.csv', index=False)