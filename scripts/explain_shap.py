import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
# from rdkit.Chem import Draw, AllChem
# from rdkit import Chem, rdBase

import seaborn as sns
import numpy as np

from src.model import BaselineAqueousModel
from src.explainer import ColorMapper, plot_weighted_molecule, make_div_legend
from sklearn import linear_model
from nemo_src.regex_tokenizer import RegExTokenizer
import shap
import pickle
import hydra
from omegaconf import OmegaConf, DictConfig


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
            [1] * len(token_id)) + ([0] * (pad_length - len(token_id)))
        token_id = torch.tensor(token_id, dtype=torch.int64).cuda()
        encoder_mask = torch.tensor(encoder_mask,
                                    dtype=torch.int64,
                                    device=token_id.device)

        return token_id, encoder_mask

    def __call__(self, text):
        token_ids, token_masks = self.tokenize_one(text)
        return {'input_ids': token_ids.tolist(),
                'input_masks': token_masks}


@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def explain_shap(cfg: DictConfig) -> None:

    cfg = OmegaConf.load('./params.yaml')
    print('SHAP EXPLAIN CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))
    cfg.model.n_batch = 32

    pl.seed_everything(cfg.model.seed)
    root = f"./data/{cfg.task.task}/{cfg.split.split}"
    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)
    test_loader = DataLoader(test, batch_size=cfg.model.n_batch,
                             shuffle=False, num_workers=8)

    basepath = f"./out/{cfg.task.task}/{cfg.split.split}"
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    ckpt_path = f"{basepath}/{mdir}/best.pt"

    if cfg.model.model in ['mmb-avg', 'mmb-ft-avg']:
        model = BaselineAqueousModel(head=cfg.head.head,
                                     finetune=cfg.model.finetune)
        model.head.load_state_dict(torch.load(ckpt_path))
    else:
        raise NotImplementedError

    if cfg.model.finetune or 'ft' in cfg.model.model:
        mmb_path = f"{basepath}/{mdir}/best_mmb.pt"
        model.mmb.load_state_dict(torch.load(mmb_path))

    ###################################

    # trainer = pl.Trainer(
    #     accelerator='gpu',
    #     gpus=1,
    #     precision=16,
    # )
    # all = trainer.predict(model, test_loader)
    # preds = [f.get('preds') for f in all]
    # labels = [f.get('labels') for f in all]
    #
    # yhat = torch.concat(preds)
    # y = torch.concat(labels)
    #
    # mse = nn.MSELoss()(yhat, y)
    # mae = nn.L1Loss()(yhat, y)
    # rmse = torch.sqrt(mse)
    #
    # data = pd.DataFrame({'y': y, 'yhat': yhat})
    # reg = linear_model.LinearRegression()
    # reg.fit(yhat.reshape(-1, 1), y)
    # slo = f"{reg.coef_[0]:.3f}"
    #
    # # text formatting for plot
    # split = f"{int(round(1.-cfg.split.split_frac, 2)*100)}% "
    # color = cfg.split.color
    # _acc = cfg.split.split
    #
    # # plot a hexagonal parity plot
    # ally = np.array(torch.concat([y, yhat], axis=0))
    # lim = [np.floor(np.min(ally)), np.ceil(np.max(ally))]
    # print('lim', lim)
    # p = sns.jointplot(x=y, y=yhat, kind='hex', color=color,
    #                   xlim=lim, ylim=lim)
    # sns.regplot(x="yhat", y="y", data=data, ax=p.ax_joint,
    #             color='grey', ci=None, scatter=False)
    # p.fig.suptitle(f"{cfg.task.plot_title} parity plot \
    #     \n{_acc} {split}test set")
    # p.set_axis_labels(f"Experimental {cfg.task.plot_propname}",
    #                   f"Model {cfg.task.plot_propname}")
    #
    # p.fig.subplots_adjust(top=0.95)
    # p.fig.tight_layout()
    # txt = f"RMSE = {rmse:.3f} \nMAE = {mae:.3f} \nn = {len(y)} \nSlope = {slo}"
    # plt.text(lim[1], lim[0],
    #          txt, ha="right", va="bottom", fontsize=14)
    # p.savefig(f"{basepath}/{mdir}/parity_plot.png")

    ###################################

    tokenizer = ShapTokenizer()
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        model,
        masker
    )

    attributions = pd.DataFrame(columns=[
        'smiles', 'tokens', 'preds', 'labels', 'atom_weights', 'split'
    ])

    coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    cmapper = ColorMapper(diverging=True, cmap=coolwarm)
    pos_cmapper = ColorMapper(color='blue')
    neg_cmapper = ColorMapper(color='red')

    make_div_legend()
    xai = cfg.model.model
    # weights = model.head.fc1.weight[0].cpu().detach().numpy()
    # weights = weights[:, None]
    # bias = model.head.fc1.bias[0].cpu().detach().numpy()

    for b_nr, batch in enumerate(test_loader):
        smiles, labels = batch
        atom_weights = []

        shapvals = explainer(smiles).values
        tokens = [tokenizer.text_to_tokens(s) for s in smiles]
        preds = model(smiles).cpu().detach().numpy()
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
            uid = b_nr * cfg.model.n_batch + b_ix
            shapval = shapvals[b_ix]
            print(uid, 'shapval:', shapval)

            atom_weight = cmapper(shapval, token)
            atom_weights.append(atom_weight)
            atom_color = cmapper.to_rdkit_cmap(atom_weight)

            pos_mask = np.where(np.sign(shapval) == 1, 1, 0)
            shap_pos = shapval * pos_mask
            pos_color = cmapper(shap_pos, token)
            pos_color = pos_cmapper.to_rdkit_cmap(pos_color)

            neg_mask = np.where(np.sign(shapval) == -1, 1, 0)
            shap_neg = shapval * neg_mask
            neg_color = cmapper(shap_neg, token)
            neg_color = neg_cmapper.to_rdkit_cmap(neg_color)

            if uid not in []:  # 17, 39, 94, 210, 217
                # segmentation fault, likely due to weird structure?
                plot_weighted_molecule(atom_color, smi, token, lab, pred,
                    f"{uid}_{xai}", f"{basepath}/{mdir}/viz/")
                # plot_weighted_molecule(pos_color, smi, token, lab, pred,
                #     f"{uid}_pos_{xai}", f"{basepath}/{mdir}/viz/")
                # plot_weighted_molecule(neg_color, smi, token, lab, pred,
                #     f"{uid}_neg_{xai}", f"{basepath}/{mdir}/viz/")

        ###############################

        print(len(atom_weights))
        print(len(shapvals))
        print(len(preds))
        res = pd.DataFrame({
            'smiles': smiles,
            'tokens': tokens,
            'preds': preds,
            # 'labels': labels,
            'shap_weights': shapvals,
            'atom_weights': atom_weights,
            'split': 'test'
            })
        attributions = pd.concat([attributions, res], axis=0)

    print('token/shap equal len', all([len(t) == len(s) for t, s in zip(
        attributions.tokens, attributions.shap_weights
    )]))

    attributions = attributions.reset_index(drop=True).rename(columns={'index': 'uid'})
    attributions.to_csv(f"{basepath}/{mdir}/attributions.csv", index=False)
    # results = results.reset_index(drop=True)
    # results = results.reset_index().rename(columns={'index':'uid'})
    # results.to_csv('/workspace/results/shap/AqueousSolu_SHAP.csv', index=False)


if __name__ == "__main__":
    explain_shap()
