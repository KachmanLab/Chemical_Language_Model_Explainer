import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from rdkit import Chem, rdBase
from rdkit.Chem import Draw, AllChem

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import pandas as pd
import os
import glob
import json
import itertools

from src.dataloader import CombiSoluDataset
from src.model import CombiRegModel
from src.explainer import ColorMapper

with open('/workspace/scripts/combi_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

test_dataset = CombiSoluDataset('/workspace/data/CombiSolu-Exp.csv', 'test',
    cfg['temp_test'], cfg['split'], data_seed=cfg['seed'], scale_logS=cfg['scale_logS'])
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'],
    shuffle=False, num_workers=8)

# logS is scaled to [0, 1] for stability, so we need to unscale it for plotting
unscale = test_dataset.unscale if cfg['scale_logS'] else lambda x: x

subfolders = [f.path for f in os.scandir('/workspace/results/combi-solu/') if f.is_dir()]
subfolders = max(subfolders, key=os.path.getmtime)
# subfolders = '/workspace/results/combi-solu/xxxxxxxx' # pick a specific run / ckpt

ckpt_path = glob.glob(os.path.join(f'{subfolders}/checkpoints/', "*"))[0]
print(subfolders, ckpt_path)

combi_model = CombiRegModel()
combi_model.load_from_checkpoint(ckpt_path)
combi_model.mmb.unfreeze()

trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    precision=16,
)
# predict with trained model (ckpt_path)
all = trainer.predict(combi_model, test_loader, ckpt_path=ckpt_path)

solu_smiles = [f.get('solu_smi') for f in all]
solv_smiles = [f.get('solv_smi') for f in all]
tokens = [f.get('tokens') for f in all]
preds = [f.get('preds') for f in all]
labels = [f.get('labels') for f in all]
masks = [f.get('masks') for f in all]
rel_weights = [f.get('rel_weights') for f in all]
atom_colors = [f.get('atom_colors') for f in all]

###################################
# parity plot

yhat = unscale(torch.concat(preds))
y = unscale(torch.concat(labels))
# print(yhat.min(), yhat.max(), y.min(), y.max())

mse = nn.MSELoss()(yhat, y)
mae = nn.L1Loss()(yhat, y)
rmse = torch.sqrt(mse)

data = pd.DataFrame({'y': y, 'yhat': yhat})
reg = linear_model.LinearRegression()
reg.fit(yhat.reshape(-1,1), y)
slo = f"{reg.coef_[0]:.3f}"

# text formatting for plot
split = f"{int(round(1.-cfg['split'], 2)*100)}%"
_tmp = f"{'T=298K' if cfg['temp_test'] else f'random {split}'}"
tmp = f"{'temp' if cfg['temp_test'] else 'rnd'}"

soluSmi = list(itertools.chain(*solu_smiles))
solvSmi = list(itertools.chain(*solv_smiles))

nSolu = f"{len(np.unique(soluSmi))} solutes"
nSolv = f"{len(np.unique(solvSmi))} solvents"

p = sns.jointplot(x='y', y='yhat', data=data, kind='hex', 
        color='b' if cfg["temp_test"] else 'g',
        xlim=[-6, 1.5], ylim=[-6, 1.5]) 
sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint, 
        color='grey', ci=None, scatter=False)

_units = '298K [mol/L]' if cfg['temp_test'] else '[mol/L]'
p.fig.suptitle(f"CombiSolu-Exp parity plot\n{_tmp} test set, <REG>,<SEP> model")
p.set_axis_labels(f"Experimental log(S), {_units}", f"Model log(S), {_units}")
p.fig.subplots_adjust(top=0.95)
p.fig.tight_layout()
txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nSlope = {slo} \nn = {len(y)} \n{nSolu}, {nSolv} "
plt.text(1.5, -5.75,
         txt, ha="right", va="bottom", fontsize=14)
p.savefig(f'/workspace/results/combi-solu/CombiSolu_parity_{tmp}.png')

###################################
#molecule visualizations

def plot_weighted_molecule_pair(
    atom_colors, solu_smi, solv_smi, tokens, logS, pred, prefix="",
):
    atom_colors = atom_colors
    bond_colors = {}
    h_rads = {} #?
    h_lw_mult = {} #?

    smiles = solu_smi + '.' + solv_smi
    label = f'Exp logS: {logS:.2f}, predicted: {pred:.2f}\nSolute: {solu_smi}\nSolvent: {solv_smi}'

    mol = Chem.MolFromSmiles(smiles)
    mol = Draw.PrepareMolForDrawing(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(550, 520)#, 700, 700)
    d.drawOptions().padding = 0.0  # No extra whitespace

    if int(mol.GetNumAtoms()) != len(atom_colors.keys()):
        cnt = int(mol.GetNumAtoms()) - len(atom_colors.keys())
        print(f"Warning: {cnt} count mismatch for {tokens}. {smiles}")
    d.DrawMoleculeWithHighlights(mol, label, atom_colors, 
        bond_colors, h_rads, h_lw_mult, -1)
    d.FinishDrawing()

    with open(file=f'/workspace/results/combi-solu/{prefix}_MolViz.png',
        mode = 'wb') as f:
        f.write(d.GetDrawingText())


b_nr = [0]*cfg['n_batch']
b_ix = list(range(cfg['n_batch']))

# find smiles string indices
target_smiles = ['C1=CC=C2C(=C1)C(=NO2)CS(=O)(=O)N',
                 'COO',
                 'CC1=CC=CC=C1']
for nr in range(len(solu_smiles)):
    for ix in range(len(solu_smiles[nr])):
        solu = solu_smiles[nr][ix]
        if solu in target_smiles:
            b_nr.append(nr)
            b_ix.append(ix)

for b_nr, b_ix in zip(b_nr, b_ix):
    token = tokens[b_nr][b_ix][1:]
    mask = masks[b_nr][b_ix]
    solu_smi = solu_smiles[b_nr][b_ix]
    solv_smi = solv_smiles[b_nr][b_ix]
    lab = unscale(labels[b_nr][b_ix])
    pred = unscale(preds[b_nr][b_ix])
    atom_color = atom_colors[b_nr][b_ix]
    plot_weighted_molecule_pair(
        atom_color, solu_smi, solv_smi, token, lab, pred, f"{b_nr}_{b_ix}"
    )