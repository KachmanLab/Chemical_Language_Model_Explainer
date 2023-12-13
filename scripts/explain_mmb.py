import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from rdkit.Chem import Draw, AllChem
from rdkit import Chem, rdBase
from itertools import chain
import seaborn as sns
import pandas as pd
import pickle
from sklearn import linear_model

from src.model import AqueousRegModel, BaselineAqueousModel
from src.maskedhead import MaskedLinearRegressionHead
from src.explainer import ColorMapper, MolecularSelfAttentionViz
import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def explain_mmb(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.load('./params.yaml')
    print('EXPLAIN CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    root = f"./data/{cfg.task.task}/{cfg.split.split}"
    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)

    # test.smiles = test.smiles[[2, 5, 12, 16]]
    # test.labels = test.labels[[2, 5, 12, 16]]
    test.smiles = [test.smiles[i] for i in [2, 5, 13, 15]]
    test.labels = [test.labels[i] for i in [2, 5, 13, 15]]
    # test.smiles = test.smiles[:16]
    # test.labels = test.labels[:16]
    test_loader = DataLoader(test, batch_size=cfg.model.n_batch,
                             shuffle=False, num_workers=8)

    basepath = f"./out/{cfg.task.task}/{cfg.split.split}"
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    ckpt_path = f"{basepath}/{mdir}/best.pt"

    head = cfg.head.head
    print(head)
    # if cfg.xai.mask:
    #     if head == 'lin_mask' or head == 'lin':
    #         head = 'lin_mask'   # MaskedLinearRegressionHead()
    #     elif head == 'hier_mask' or head == 'hier':
    #         head = 'hier_mask'  # MaskedRegressionHead()
    if cfg.model.model == 'mmb':
        model = AqueousRegModel(head=head,
                                finetune=cfg.model.finetune)
        model.head.load_state_dict(torch.load(ckpt_path))
        xai = 'mmb'
    if cfg.model.finetune or cfg.model.model == 'mmb-ft':
        model = AqueousRegModel(head=head,
                                finetune=cfg.model.finetune)
        # model = model.load_from_checkpoint(ckpt_path, head=head)
        mmb_path = f"{basepath}/{mdir}/best_mmb.pt"
        model.mmb.load_state_dict(torch.load(mmb_path))
        model.head.load_state_dict(torch.load(ckpt_path))
        xai = 'mmb-ft'
    elif cfg.model.model == 'mmb-avg':
        print('wrong xai config: mmb-avg+explain_mmb should be shap')
        raise NotImplementedError

    model.mmb.unfreeze()
    model.eval()

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

    attributions = pd.DataFrame({
        "smiles": list(chain(*smiles)),
        "tokens": list(chain(*tokens)),
        "atom_weights": list(chain(*rel_weights)),
        "rdkit_colors": list(chain(*rdkit_colors)),
        "preds": torch.concat([f.get('preds') for f in all]).cpu().numpy(),
        # "labels": torch.concat([f.get('labels') for f in all]).cpu().numpy(),
        'split': 'test'
    })

    # <pos>,<neg> attribution for mmb-ft+lin
    sign_weights, sign_colors = {}, {}
    if cfg.model.model == 'mmb-ft' and 'lin' in cfg.head.head:
        for sign in ['pos', 'neg']:
            # change head to masked head variant & load
            model.head = MaskedLinearRegressionHead(sign=sign)
            model.head.load_state_dict(torch.load(ckpt_path))
            model.eval()
            model.explainer = MolecularSelfAttentionViz(sign=sign)

            # change viz color to red/blue
            # color = 'blue' if sign == 'pos' else 'red'
            # model.cmapper = ColorMapper(color=color)

            all_sgn = trainer.predict(model, test_loader)
            sign_weights[sign] = [f.get('rel_weights') for f in all_sgn]
            sign_colors[sign] = [f.get('rdkit_colors') for f in all_sgn]
            sign_weights[f'{sign}_preds'] = [f.get('preds') for f in all_sgn]

            # should _not_ be equal if applying sign_mask on fwd()
            # sign_preds = [f.get('preds') for f in all_sgn]
            # assert torch.allclose(torch.concat(preds), torch.concat(sign_preds),
            #                       atol = 1e-2)

            attributions[f"{sign}_weights"] = list(chain(*sign_weights[sign]))
            attributions[f"{sign}_colors"] = list(chain(*sign_colors[sign]))
    # </pos>,</neg>

    attributions = attributions.reset_index().rename(columns={'index': 'uid'})
    attributions.to_csv(f"{basepath}/{mdir}/attributions.csv", index=False)

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
    ally = np.array(torch.concat([y, yhat], axis=0))
    lim = [np.floor(np.min(ally)), np.ceil(np.max(ally))]
    print('lim', lim)
    p = sns.jointplot(x=y, y=yhat, kind='hex', color=color,
                      xlim=lim, ylim=lim)
    sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint,
                color='grey', ci=None, scatter=False)
    p.fig.suptitle(f"{cfg.task.plot_title} parity plot \
        \n{_acc} {split}test set")
    p.set_axis_labels(f"Experimental {cfg.task.plot_propname}",
                      f"Model {cfg.task.plot_propname}")

    p.fig.subplots_adjust(top=0.95)
    p.fig.tight_layout()
    txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo}"
    plt.text(lim[1], -lim[0],
             txt, ha="right", va="bottom", fontsize=14)
    p.savefig(f"{basepath}/{mdir}/parity_plot.png")

    ###################

    def plot_weighted_molecule(
                atom_colors, smiles, token, label, pred, prefix=""
            ):
        atom_colors = atom_colors
        bond_colors = {}
        h_rads = {}  # ?
        h_lw_mult = {}  # ?

        label = f'Exp {cfg.task.plot_propname}: {label:.2f}, predicted: {pred:.2f}\n{smiles}'

        mol = Chem.MolFromSmiles(smiles)
        mol = Draw.PrepareMolForDrawing(mol)
        d = Draw.rdMolDraw2D.MolDraw2DCairo(700, 700)
        d.drawOptions().padding = 0.0

        # some plotting issues for 'C@@H' and 'C@H' tokens since
        # another H atom is rendered explicitly.
        # Might break for ultra long SMILES using |c:1:| notation
        vocab = model.cmapper.atoms + model.cmapper.nonatoms

        mismatch = int(mol.GetNumAtoms()) - len(atom_colors.keys())
        # if mismatch != 0:
        if mismatch < 0:
            print(f"Warning: {mismatch}: \
                 {[t for t in token if t not in vocab]}, {prefix}")
            # print(f"count mismatch for {smiles}:\
                 # {[t for t in token if t not in vocab]}")
            # print(f'{token}')
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
    # fid = model.head.fids

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

            # if uid not in [39, 94, 170, 210, 217, 451, 505, 695, 725, 755]:
                # segmentation fault, likely due to weird structure?
            if uid not in []:
                plot_weighted_molecule(
                    atom_color, smi, token, lab, pred, f"{uid}_{xai}"
                )
            if sign_weights:
                pos_color = sign_colors['pos'][b_nr][b_ix]
                neg_color = sign_colors['neg'][b_nr][b_ix]
                pos_pred = sign_weights[f'pos_preds'][b_nr][b_ix]
                neg_pred = sign_weights[f'neg_preds'][b_nr][b_ix]
                assert pos_pred + neg_pred - pred <= 5e-2
                plot_weighted_molecule(
                    pos_color, smi, token, lab, pos_pred, f"{uid}_pos_{xai}"
                )
                plot_weighted_molecule(
                    neg_color, smi, token, lab, neg_pred, f"{uid}_neg_{xai}"
                )
        if b_nr > 2:
            break

if __name__ == "__main__":
    explain_mmb()
