import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rdkit.Chem import Draw, AllChem, rdMolDescriptors
from rdkit import Chem, rdBase
import itertools
import seaborn as sns
import numpy as np
import pandas as pd
import os
import glob
import json
from sklearn import linear_model

from src.prop_loader import LogPDataset
from src.model import AqueousRegModel, BaselineAqueousModel
from src.explainer import ColorMapper, plot_weighted_molecule 
from nemo_src.regex_tokenizer import RegExTokenizer
import shap

with open('/workspace/scripts/logp_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])
test_dataset = LogPDataset('/workspace/data/opera_logp.csv', 'test',
    cfg['split'], data_seed=cfg['seed'])
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

subfolders = [f.path for f in os.scandir('/workspace/results/logp/models/') \
    if (f.path.endswith('.pt') and ('shap' in os.path.split(f)[1]))]
ckpt_path = max(subfolders, key=os.path.getmtime)

ft_model = BaselineAqueousModel().load_from_checkpoint(ckpt_path)
xai = f"shap"

ft_model.mmb.unfreeze()
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

results = pd.DataFrame(
    columns=['smiles', 'tokens', 'logp_pred', 'logp_exp', 'shap_raw', 
    'shap_weights', 'shap_colors', 'split']
)
cmapper = ColorMapper()

for batch in test_loader:
    smiles, labels = batch
    shap_vals = explainer(smiles).values
    tokens = [tokenizer.text_to_tokens(s) for s in smiles]
    preds = ft_model(smiles).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    shap_raw = [cmapper.filter_atoms(v,t) for v,t in zip(shap_vals, tokens)]
    shap_weights = [cmapper(v,t) for v,t in zip(shap_vals, tokens)]
    shap_colors = [cmapper.to_rdkit_cmap(x) for x in shap_weights]

    ###############################
    for b_ix in range(len(smiles)):
        token = tokens[b_ix]
        smi = smiles[b_ix]
        lab = labels[b_ix]
        pred = preds[b_ix]
        shap_color = shap_colors[b_ix]
        uid = len(results) + b_ix
        
        print(uid)
        if uid not in [252]:
            try:
                plot_weighted_molecule(shap_color, smi, token, lab, pred, 
                f"{uid}_shap", f'/workspace/results/logp/viz_shap')
            except:
                print(f'** {uid} failed to plot')
    ###############################
    
    res = pd.DataFrame({
        'smiles': smiles,
        'tokens': tokens,
        'logp_pred': preds,
        'logp_exp': labels,
        'shap_raw': shap_raw,
        'shap_weights': shap_weights,
        'shap_colors': shap_colors,
        'split': 'test'
        })
    print(res)
    results = pd.concat([results, res], axis=0)

print('token/shap equal len', all([len(t) == len(s) for t, s in zip(
    results.tokens, results.shap_weights
)]))

results = results.reset_index(drop=True)
results = results.reset_index().rename(columns={'index':'uid'})
results.to_csv('/workspace/results/logp/logp_shap_predictions.csv', index=False)


# PLOT
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'],                                   
    shuffle=False, num_workers=8)                                                                   
                                                                                                    
trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    precision=16,
)

all = trainer.predict(ft_model, test_loader, ckpt_path=ckpt_path)

preds = [f.get('preds') for f in all]
labels = [f.get('labels') for f in all]

# load data and calculate errors
yhat = torch.concat(preds)
y = torch.concat(labels)

mse = nn.MSELoss()(yhat, y)
mae = nn.L1Loss()(yhat, y)
rmse = torch.sqrt(mse)
                                                                                                    
data = pd.DataFrame({'y': y, 'yhat': yhat})
reg = linear_model.LinearRegression()
reg.fit(yhat.reshape(-1,1), y)
slo = f"{reg.coef_[0]:.3f}"
                                                                          
# text formatting for plot
split = f"{int(round(1.-cfg['split'], 2)*100)}%"
# plot a hexagonal parity plot
p = sns.jointplot(x=y, y=yhat, kind='hex', color='g',
                    xlim=[-4, 6.5], ylim=[-4, 6.5])
sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint, color='grey', ci=None,                       
            scatter=False)
p.fig.suptitle(f"logP parity plot: SHAP + average-pooling")
p.set_axis_labels('Experimental log(P)', 'Model log(P)')
p.fig.subplots_adjust(top=0.95)
p.fig.tight_layout()
txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo} "                        
plt.text(6, -4.,
        txt, ha="right", va="bottom", fontsize=14)
p.savefig(f'/workspace/results/logp/logp_parity_plot_{xai}.png')