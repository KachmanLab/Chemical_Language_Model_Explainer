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

with open('/workspace/scripts/logp_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])
test_dataset = LogPDataset('/workspace/data/logp_dataset.csv', 'test',
    cfg['split'], data_seed=cfg['seed'])
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)
    
subfolders = [f.path for f in os.scandir('/workspace/results/logp/models/') \
    if (f.path.endswith('.pt') or f.path.endswith('.ckpt'))]
ckpt_path = max(subfolders, key=os.path.getmtime)

if cfg['model'] == 'logp':
    ft_model = AqueousRegModel()
    xai = f"logp"
elif cfg['model'] == 'shap':
    raise NotImplementedError
    ft_model = BaselineAqueousModel()
    xai = f"shap"

ft_model = ft_model.load_from_checkpoint(ckpt_path)
ft_model.mmb.unfreeze()

trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    precision=16,
)

# predict with trained model (ckpt_path)
all = trainer.predict(ft_model, test_loader, ckpt_path=ckpt_path)

smiles = [f.get('smiles') for f in all]
tokens = [f.get('tokens') for f in all]
preds = [f.get('preds') for f in all]
labels = [f.get('labels') for f in all]
masks = [f.get('masks') for f in all]

if cfg['model'] == 'logp':
    rel_weights = [f.get('rel_weights') for f in all]
    rdkit_colors = [f.get('rdkit_colors') for f in all]
elif cfg['model'] == 'shap':
    raise NotImplementedError
    # shap_colors = [f.get('shap_colors') for f in all]

##################################
cmapper = ColorMapper()
def calc_crippen(smi):
    mol = Chem.MolFromSmiles(smi)
    desc = rdMolDescriptors._CalcCrippenContribs(mol)
    logp = np.array(desc)[:, 0]
    return logp

allsmiles = list(itertools.chain(*smiles))
crippen = [calc_crippen(smi) for smi in allsmiles]
crippen_preds = [sum(x) for x in crippen]
crippen_colors = [
    cmapper.to_rdkit_cmap(Normalize()(x)) for x in crippen
]

###################################

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
                 xlim=[-4, 5.], ylim=[-4, 5.])
sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint, color='grey', ci=None, scatter=False)
p.fig.suptitle(f"logP parity plot")
p.set_axis_labels('Experimental log(P)', 'Model log(P)')
p.fig.subplots_adjust(top=0.95)
p.fig.tight_layout()
txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo} "
plt.text(4, -4.,
         txt, ha="right", va="bottom", fontsize=14)
p.savefig(f'/workspace/results/logp/logp_parity_plot_{xai}.png')

###################################
# CRIPPEN
yhat = torch.tensor(crippen_preds)

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
                 xlim=[-4, 5.], ylim=[-4, 5.])
sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint, color='grey', ci=None, scatter=False)
p.fig.suptitle(f"Crippen's logP prediction parity plot")
p.set_axis_labels('Experimental log(P)', 'Model log(P)')
p.fig.subplots_adjust(top=0.95)
p.fig.tight_layout()
txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo} "
plt.text(4, -4.,
         txt, ha="right", va="bottom", fontsize=14)
p.savefig(f'/workspace/results/logp/logp_parity_plot_crippen.png')

###################################
# plot entire test set:
b_indices = list(range(cfg['n_batch']))
for b_nr, _ in enumerate(all):
    for b_ix in range(len(smiles[b_nr])):
        token = tokens[b_nr][b_ix]
        mask = masks[b_nr][b_ix]
        smi = smiles[b_nr][b_ix]
        lab = labels[b_nr][b_ix]
        pred = preds[b_nr][b_ix]
        uid = b_nr * cfg['n_batch'] + b_ix
        crippen_color = crippen_colors[uid]
        crippen_pred = crippen_preds[uid]

        if cfg['model'] == 'logp':
            atom_color = rdkit_colors[b_nr][b_ix]
        elif cfg['model'] == 'shap':
            raise NotImplementedError
            atom_color = salience_colors[b_nr][b_ix]
        
        # segmentation fault, likely due to weird structure?
        if uid not in [39, 94, 210, 217, 252]:
            plot_weighted_molecule(atom_color, smi, token, lab, pred, 
                f"{uid}_{xai}", f'/workspace/results/logp/viz_ours')

            plot_weighted_molecule(crippen_color, smi, token, lab, crippen_pred, 
                f"{uid}_crip", f'/workspace/results/logp/viz_crippen')