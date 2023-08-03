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

from src.dataloader import AqSolDataset
from src.model import AqueousRegModel, BaselineAqueousModel
from src.explainer import ColorMapper 
from nemo_src.regex_tokenizer import RegExTokenizer
import shap
from typing import List

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)
    
subfolders = [f.path for f in os.scandir('/workspace/results/aqueous/models/') \
    if (f.path.endswith('.pt') or f.path.endswith('.ckpt'))]
ckpt_path = max(subfolders, key=os.path.getmtime)

ft_model = BaselineAqueousModel().load_from_checkpoint(ckpt_path)
xai = f"shap"
ft_model.unfreeze()
mmb_tokenizer = ft_model.tokenizer
print(mmb_tokenizer)

def shap_predict(input):
    tokens, masks = ft_model.tokenizer.tokenize(input)
    preds = ft_model.mask_forward(tokens, masks)
    return preds.cpu().detach().numpy()


# masker = shap.maskers.Text(ft_model)
class ShapTokenizer(RegExTokenizer):
    """This minimal subset means the tokenizer must return a 
    dictionary with ‘input_ids’ and then either include an 
    ‘offset_mapping’ entry in the same dictionary or provide 
    a .convert_ids_to_tokens or .decode method."""

    def __init__(self):
        super().__init__()
        self.load_tokenizer()
        # self._update_cache()
        # print(self.vocab)
        print(self._decode_vocab)

    def convert_ids_to_tokens(self, ids):
        return self.ids_to_tokens(ids)

    # def tokenize(self, smis: List[str]):
    #     tokens = [self.text_to_tokens(s) for s in smis]
    #     token_ids = [self.token_to_ids(t) for t in tokens]
    #     pad_length = max([len(seq) for seq in token_ids])
        
    #     encoder_masks = [
    #         ([1] * len(seq)) + ([0] * (pad_length - len(seq))) \
    #             for seq in token_ids
    #     ]
    #     token_ids = [
    #         seq + ([self.pad_id] * (pad_length - len(seq))) \
    #             for seq in token_ids 
    #     ]

    #     token_ids = torch.tensor(token_ids, dtype=torch.int64).cuda()
    #     encoder_masks = torch.tensor(encoder_masks,
    #                                 dtype=torch.int64,
    #                                 device=token_ids.device)

    #     return token_ids, encoder_masks


    def tokenize_one(self, smi: str):
        token = self.text_to_tokens(smi) 
        token_id = self.token_to_ids(token)
        print('**', token_id)

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
        # input_ids = self.tokens_to_ids(tokens, self.vocab)
        return {'input_ids': token_ids.tolist(),
                'input_masks': token_masks}

tokenizer = ShapTokenizer()
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(
    ft_model,
    # shap_predict, 
    # mmb_tokenizer, 
    masker
    # output_names=labels
)


smi3 = ['CCCCC1=C(C)N=C(NCC)NC1=O', 'Oc(cccc2)c2c1ccccc1', 'NC(OCC)=O']
smi1 = ['NC(OCC)=O'] 
smirep = ['NC(OCC)=O', 'NC(OCC)=O', 'NC(OCC)=O', 'NC(OCC)=O', 'NC(OCC)=O']

shap_values = explainer(smi3)
print(shap_values)
shap.plots.text(shap_values)


# [279, 272, 275, 285, 272, 272, 281, 280, 285]
# array([[ 0.02033204,  0.02452358, -0.02023098,  0.03454226,  0.01868196,
#          0.0447597 ,  0.03440047, -0.02958229,  0.01401009]