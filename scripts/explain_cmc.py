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
import json
from sklearn import linear_model

from src.prop_loader import CMCDataset
from src.model import AqueousRegModel, BaselineAqueousModel
from src.explainer import ColorMapper, plot_weighted_molecule

with open('/workspace/scripts/cmc_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])
test_dataset = CMCDataset('/workspace/data/cmcdata.csv', 'test',
                cfg['split_type'], cfg['split'], data_seed=cfg['seed'])
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'],
                         shuffle=False, num_workers=8)

subfolders = [f.path for f in os.scandir('/workspace/results/cmc/models/')
    if (f.path.endswith('.pt') and ('cmc' in os.path.split(f)[1]))]
print(subfolders)
ckpt_path = max(subfolders, key=os.path.getmtime)

if cfg['model'] == 'mmb':
    model = AqueousRegModel(head=cfg['head'])
    xai = f"mmb"
elif cfg['model'] == 'shap':
    raise NotImplementedError
    # model = BaselineAqueousModel()
    # xai = f"shap"

model = model.load_from_checkpoint(ckpt_path)
model.mmb.unfreeze()

trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    precision=16,
)

# predict with trained model (ckpt_path)
all = trainer.predict(model, test_loader, ckpt_path=ckpt_path)

smiles = [f.get('smiles') for f in all]
tokens = [f.get('tokens') for f in all]
preds = [f.get('preds') for f in all]
labels = [f.get('labels') for f in all]
masks = [f.get('masks') for f in all]

if cfg['model'] == 'mmb':
    # raw_weights = [f.get('rel_weights') for f in all]
    atom_weights = [f.get('atom_weights') for f in all]
    rdkit_colors = [f.get('rdkit_colors') for f in all]
elif cfg['model'] == 'shap':
    raise NotImplementedError
    # shap_colors = [f.get('shap_colors') for f in all]

# print(type(rel_weights[0][0]))
print(type(atom_weights[0][0]))
print(atom_weights[0])
# print(rel_weights[0])
print(atom_weights[0][0].shape)
# print(rel_weights[0][0].shape)

##################################
cmapper = ColorMapper()
def calc_crippen(smi):
    mol = Chem.MolFromSmiles(smi)
    desc = rdMolDescriptors._CalcCrippenContribs(mol)
    cmc = np.array(desc)[:, 0]
    return cmc

allsmiles = list(itertools.chain(*smiles))
#crippen = [calc_crippen(smi) for smi in allsmiles]
#crippen_preds = [sum(x) for x in crippen]
#crippen_norm = [Normalize()(x) for x in crippen]
#crippen_colors = [
#    cmapper.to_rdkit_cmap(x) for x in crippen_norm
#]

###################################

# load data and calculate errors
yhat = torch.concat(preds)
y = torch.concat(labels)

mse = nn.MSELoss()(yhat, y)
mae = nn.L1Loss()(yhat, y)
rmse = torch.sqrt(mse)

data = pd.DataFrame({'y': y, 'yhat': yhat})
reg = linear_model.LinearRegression()
reg.fit(yhat.reshape(-1, 1), y)
slo = f"{reg.coef_[0]:.3f}"

# text formatting for plot
split = f"{int(round(1.-cfg['split'], 2)*100)}%"
# plot a hexagonal parity plot
p = sns.jointplot(x=y, y=yhat, kind='hex', color='g', xlim=[-6, 0], ylim=[-6, 0])
sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint, color='grey', ci=None, scatter=False)
p.fig.suptitle(f"p(CMC) parity plot: MegaMolBART + <REG> token head")
p.set_axis_labels('Experimental p(CMC)', 'Model p(CMC)')
p.fig.subplots_adjust(top=0.95)
p.fig.tight_layout()
txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo} "
plt.text(-0, -6.,
         txt, ha="right", va="bottom", fontsize=14)
p.savefig(f'/workspace/results/cmc/cmc_parity_plot_{xai}.png')

###################################
# # CRIPPEN
# yhat = torch.tensor(crippen_preds)

# mse = nn.MSELoss()(yhat, y)
# mae = nn.L1Loss()(yhat, y)
# rmse = torch.sqrt(mse)

# data = pd.DataFrame({'y': y, 'yhat': yhat})
# reg = linear_model.LinearRegression()
# reg.fit(yhat.reshape(-1,1), y)
# slo = f"{reg.coef_[0]:.3f}"

# # text formatting for plot
# split = f"{int(round(1.-cfg['split'], 2)*100)}%"
# # plot a hexagonal parity plot
# p = sns.jointplot(x=y, y=yhat, kind='hex', color='g',
#                  xlim=[-4, 6.5], ylim=[-4, 6.5])
# sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint, color='grey',
#     ci=None, scatter=False)
# p.fig.suptitle(f"Crippen's logP prediction parity plot")
# p.set_axis_labels('Experimental log(P)', 'Model log(P)')
# p.fig.subplots_adjust(top=0.95)
# p.fig.tight_layout()
# txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo} "
# plt.text(6, -4.,
#          txt, ha="right", va="bottom", fontsize=14)
# p.savefig(f'/workspace/results/logp/logp_parity_plot_crippen.png')

##################################
## predict & write to csv for further analysis
alltokens = list(itertools.chain(*[f.get('tokens') for f in all]))
all_atom_weights = list(itertools.chain(*atom_weights))
# all_rel_weights = list(itertools.chain(*rel_weights))
all_rdkit_colors = list(itertools.chain(*rdkit_colors))
allpreds = torch.concat([f.get('preds') for f in all]).cpu().numpy()
alllabels = torch.concat([f.get('labels') for f in all]).cpu().numpy()
# print(type(all_rel_weights[0]))
print(type(all_atom_weights[0]))
# print(type(crippen[0]))
# print(type(crippen_norm[0]))

results = pd.DataFrame({
    'smiles': allsmiles,
    'tokens': alltokens,
    'cmc_pred': allpreds,
    # 'cmc_crippen': crippen_preds,
    'cmc_exp': alllabels,
    # 'ours_raw': [np.array(x) for x in all_rel_weights],
    'ours_weights': all_atom_weights,
    'ours_colors': all_rdkit_colors,
    # 'crippen_raw': crippen, 
    # 'crippen_weights': crippen_norm,
    # 'crippen_colors': crippen_colors,
    'split': 'test'
    })
# results['ours_weights'] = results['ours_weights'].astype(str)
# # results['all_rdkit_colors'] = results['all_rdkit_colors'].astype(str)
# results['crippen_raw'] = results['crippen_raw'].astype(str)
# results['crippen_weights'] = results['crippen_weights'].astype(str)
# results['crippen_colors'] = results['crippen_colors'].astype(str)

# reset index to correspond to visualization UID
results = results.reset_index(drop=True)
results = results.reset_index().rename(columns={'index':'uid'})
with np.printoptions(linewidth=9000):
    results.to_csv('/workspace/results/cmc/cmc_predictions.csv', index=False)
###################################
# plot entire test set:
print('plotting!')
b_indices = list(range(cfg['n_batch']))
for b_nr, _ in enumerate(all):
    for b_ix in range(len(smiles[b_nr])):
        token = tokens[b_nr][b_ix]
        mask = masks[b_nr][b_ix]
        smi = smiles[b_nr][b_ix]
        lab = labels[b_nr][b_ix]
        pred = preds[b_nr][b_ix]
        uid = b_nr * cfg['n_batch'] + b_ix
        # crippen_color = crippen_colors[uid]
        # crippen_pred = crippen_preds[uid]

        if cfg['model'] == 'mmb':
            atom_color = rdkit_colors[b_nr][b_ix]
        elif cfg['model'] == 'shap':
            raise NotImplementedError
            atom_color = salience_colors[b_nr][b_ix]
        
        # segmentation fault, likely due to weird structure?
        if uid not in [np.nan]:
            plot_weighted_molecule(atom_color, smi, token, lab, pred, 
                f"{uid}_{xai}", f'/workspace/results/cmc/viz_ours')

            # plot_weighted_molecule(crippen_color, smi, token, lab, crippen_pred, 
            #     f"{uid}_crip", f'/workspace/results/cmc/viz_crippen')
