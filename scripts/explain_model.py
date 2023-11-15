import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from rdkit.Chem import Draw, AllChem
from rdkit import Chem, rdBase

import seaborn as sns
import pandas as pd
import pickle
from sklearn import linear_model

from src.model import AqueousRegModel, BaselineAqueousModel
from src.explainer import ColorMapper
import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(
    version_base="1.3", config_path="/workspace/conf", config_name="config")
def explain(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.load('/workspace/params.yaml')
    print('EXPLAIN CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    root = f"/workspace/data/{cfg.task.task}/{cfg.split.split}"
    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)
    test_loader = DataLoader(test, batch_size=cfg.model.n_batch,
                             shuffle=False, num_workers=8)

    basepath = f"/workspace/out/{cfg.task.task}/{cfg.split.split}"
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    ckpt_path = f"{basepath}/{mdir}/best.pt"

    head = cfg.head.head
    if cfg.xai.mask:
        if head == 'lin_mask' or head == 'lin':
            head = 'lin_mask'   # MaskedLinearRegressionHead()
        elif head == 'hier_mask' or head == 'hier':
            head = 'hier_mask'  # MaskedRegressionHead()

    if cfg.model.model == 'mmb':
        model = AqueousRegModel(head=head)
        model.head.load_state_dict(torch.load(ckpt_path))
        xai = 'mmb'
    if cfg.model.finetune or cfg.model.model == 'mmb-ft':
        model = AqueousRegModel(head=head)
        # model = model.load_from_checkpoint(ckpt_path, head=head)
        mmb_path = f"{basepath}/{mdir}/best_mmb.pt"
        model.mmb.load_state_dict(torch.load(mmb_path))
        model.head.load_state_dict(torch.load(ckpt_path))
        xai = 'mmb-ft'
    elif cfg.model.model == 'mmb-avg':
        model = BaselineAqueousModel(head=head)
        xai = 'mmb-avg'

    model.mmb.unfreeze()

    trainer = pl.Trainer(
        accelerator='gpu',
        gpus=1,
        precision=16,
    )

    # predict with trained model (ckpt_path)
    all = trainer.predict(model, test_loader)

    smiles = [f.get('smiles') for f in all]
    tokens = [f.get('tokens') for f in all]
    preds = [f.get('preds') for f in all]
    labels = [f.get('labels') for f in all]
    # masks = [f.get('masks') for f in all]

    if cfg.model.model in ['mmb', 'mmb-ft']:
        rel_weights = [f.get('rel_weights') for f in all]
        rdkit_colors = [f.get('rdkit_colors') for f in all]
    elif cfg.model.model == 'mmb-avg':
        raise NotImplementedError
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
    reg.fit(yhat.reshape(-1, 1), y)
    slo = f"{reg.coef_[0]:.3f}"

    # text formatting for plot
    split = f"{int(round(1.-cfg.split.split_frac, 2)*100)}% "
    color = cfg.split.color
    _acc = cfg.split.split

    # plot a hexagonal parity plot
    p = sns.jointplot(x=y, y=yhat, kind='hex', color=color,
                      xlim=[-12, 2], ylim=[-12, 2])
    sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint,
                color='grey', ci=None, scatter=False)
    p.fig.suptitle(f"{cfg.task.plot_title} parity plot \
        \n{_acc} {split}test set")
    p.set_axis_labels(f"Experimental {cfg.task.plot_propname}",
                      f"Model {cfg.task.plot_propname}")

    p.fig.subplots_adjust(top=0.95)
    p.fig.tight_layout()
    txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo}"
    plt.text(2, -11.5,
             txt, ha="right", va="bottom", fontsize=14)
    p.savefig(f"{basepath}/{mdir}/parity_plot.png")

    ###################

    def plot_weighted_molecule(
                atom_colors, smiles, token, logS, pred, prefix=""
            ):
        atom_colors = atom_colors
        bond_colors = {}
        h_rads = {}  # ?
        h_lw_mult = {}  # ?

        label = f'Exp logS: {logS:.2f}, predicted: {pred:.2f}\n{smiles}'

        mol = Chem.MolFromSmiles(smiles)
        mol = Draw.PrepareMolForDrawing(mol)
        d = Draw.rdMolDraw2D.MolDraw2DCairo(700, 700)
        d.drawOptions().padding = 0.0

        # some plotting issues for 'C@@H' and 'C@H' tokens since
        # another H atom is rendered explicitly.
        # Might break for ultra long SMILES using |c:1:| notation
        vocab = model.cmapper.atoms + model.cmapper.nonatoms

        mismatch = int(mol.GetNumAtoms()) - len(atom_colors.keys())
        if mismatch != 0:
            print(f"Warning: {mismatch}")
            print(f"count mismatch for {smiles}:\
                 {[t for t in token if t  not in vocab]}")
            print(f'{token}')
            d.DrawMolecule(mol)

        else:
            d.DrawMoleculeWithHighlights(
                mol, label, atom_colors, bond_colors, h_rads, h_lw_mult, -1
            )
        # todo legend
        d.FinishDrawing()

        with open(file=f"{basepath}/{mdir}/viz/{prefix}_MolViz.png",
                  mode='wb') as f:
            f.write(d.GetDrawingText())

    ###################
    fid = model.head.fids
    # fid = 00

    # plot entire test set:
    for b_nr, _ in enumerate(all):
        for b_ix in range(len(smiles[b_nr])):
            token = tokens[b_nr][b_ix]
            # mask = masks[b_nr][b_ix]
            smi = smiles[b_nr][b_ix]
            lab = labels[b_nr][b_ix]
            pred = preds[b_nr][b_ix]
            uid = b_nr * cfg.model.n_batch + b_ix

            if cfg.model.model in ['mmb', 'mmb-ft']:
                atom_color = rdkit_colors[b_nr][b_ix]
            elif cfg.model.model == 'mmb-avg':
                atom_color = salience_colors[b_nr][b_ix]

            # print(uid)
            if uid not in [39, 94, 170, 210, 217, 451, 505, 695, 725, 755]:
                # segmentation fault, likely due to weird structure?
                plot_weighted_molecule(
                    atom_color, smi, token, lab, pred, f"{uid}_{xai}_{fid}"
                )


if __name__ == "__main__":
    explain()
