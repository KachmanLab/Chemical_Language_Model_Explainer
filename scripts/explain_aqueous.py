import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from rdkit.Chem import Draw, AllChem
from rdkit import Chem, rdBase

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import json
from sklearn import linear_model

from src.dataloader import AqSolDataset
from src.model import AqueousRegModel, BaselineAqueousModel
from src.explainer import ColorMapper 

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

subfolders = [f.path for f in os.scandir('/workspace/results/aqueous-solu/') if f.is_dir()]
subfolders = max(subfolders, key=os.path.getmtime)

if cfg['model'] == 'reg':
    # subfolders = '/workspace/results/aqueous-solu/xxxxxxx'  # pick a specific run / ckpt
    ft_model = AqueousRegModel()
    xai = f"reg"
elif cfg['model'] == 'baseline':
    # subfolders = '/workspace/results/aqueous-solu/xxxxxxx'  # pick a specific run / ckpt
    ft_model = BaselineAqueousModel()
    xai = f"sal"

ckpt_path = glob.glob(os.path.join(f'{subfolders}/checkpoints/', "*"))[0]
ft_model.load_from_checkpoint(ckpt_path)
trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    precision=16,
)

ft_model.mmb.unfreeze()
# predict with trained model (ckpt_path)
all = trainer.predict(ft_model, test_loader, ckpt_path=ckpt_path)

smiles = [f.get('smiles') for f in all]
tokens = [f.get('tokens') for f in all]
preds = [f.get('preds') for f in all]
labels = [f.get('labels') for f in all]
masks = [f.get('masks') for f in all]

if cfg['model'] == 'reg':
    rel_weights = [f.get('rel_weights') for f in all]
    atom_colors = [f.get('atom_colors') for f in all]
elif cfg['model'] == 'baseline':
    salience_colors = [f.get('salience_colors') for f in all]

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
_acc = f"{'accurate' if cfg['acc_test'] else 'random'}"
split = f"{int(round(1.-cfg['split'], 2)*100)}%"
_split = f'{split} ' if not cfg['acc_test'] else ''
acc = f"{'acc' if cfg['acc_test'] else 'rnd'}" 
# plot a hexagonal parity plot
p = sns.jointplot(x=y, y=yhat, kind='hex', color='r' if cfg["acc_test"] else 'b',
                 xlim=[-12,2], ylim=[-12, 2])
sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint, color='grey', ci=None, scatter=False)
p.fig.suptitle(f"AqueousSolu parity plot \n{_acc} {_split}test set, 298K")
p.set_axis_labels('Experimental log(S), 298K [mol/L]', 'Model log(S), 298K [mol/L]')
p.fig.subplots_adjust(top=0.95)
p.fig.tight_layout()
txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo} "
plt.text(2,-11.5,
         txt, ha="right", va="bottom", fontsize=14)
p.savefig(f'/workspace/results/aqueous-solu/Aqueous_parity_{_acc}_{xai}.png')

###################

def plot_weighted_molecule(atom_colors, smiles, token, logS, pred, prefix=""):
    atom_colors = atom_colors
    bond_colors = {}
    h_rads = {} #?
    h_lw_mult = {} #?

    label = f'Exp logS: {logS:.2f}, predicted: {pred:.2f}\n{smiles}'

    mol = Chem.MolFromSmiles(smiles)
    mol = Draw.PrepareMolForDrawing(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(700, 700)
    d.drawOptions().padding = 0.0 

    # some plotting issues for 'C@@H' and 'C@H' tokens since 
    # another H atom is rendered explicitly. 
    # Might break for ultra long SMILES using |c:1:| notation
    vocab = ft_model.cmapper.atoms + ft_model.cmapper.nonatoms
    if int(mol.GetNumAtoms()) != len(atom_colors.keys()):
        print(f"Warning: {int(mol.GetNumAtoms()) - len(atom_colors.keys())}")
        print(f"count mismatch for {smiles}:\
             {[t for t in token if t  not in vocab]}")
        print(f'{token}')

    d.DrawMoleculeWithHighlights(
        mol, label, atom_colors, bond_colors, h_rads, h_lw_mult, -1
    )
    # todo legend
    d.FinishDrawing()
    
    with open(file=f'/workspace/results/aqueous-solu/{prefix}_MolViz.png',
        mode = 'wb') as f:
        f.write(d.GetDrawingText())

###################
# plot whole batch:
b_nr = [0]*cfg['n_batch'] 
b_ix = list(range(cfg['n_batch']))

# add specific molecules:
target_smiles = ['Oc1cccc(c1)C(O)=O']

for nr in range(len(smiles)):
    for ix in range(len(smiles[nr])):
        smi = smiles[nr][ix]
        if smi in target_smiles:
            print(smi, nr, ix)
            b_nr.append(nr)
            b_ix.append(ix)

for b_nr, b_ix in zip(b_nr, b_ix):
    token = tokens[b_nr][b_ix]
    mask = masks[b_nr][b_ix]
    smi = smiles[b_nr][b_ix]
    lab = labels[b_nr][b_ix]
    pred = preds[b_nr][b_ix]
    
    if cfg['model'] == 'reg':
        atom_color = atom_colors[b_nr][b_ix]
    elif cfg['model'] == 'baseline':
        atom_color = salience_colors[b_nr][b_ix]

    plot_weighted_molecule(atom_color, smi, token, lab, pred, f"{b_nr}_{b_ix}_{xai}")