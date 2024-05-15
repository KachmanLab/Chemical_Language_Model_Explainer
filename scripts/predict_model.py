import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from src.model import AqueousRegModel, BaselineAqueousModel, ECFPLinear
from src.dataloader import ECFPDataSplit
import pickle
import hydra
import json
from omegaconf import OmegaConf, DictConfig
from sklearn.linear_model import LinearRegression
from src.explainer import ColorMapper, MolecularSelfAttentionViz
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler

@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def predict_model(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.load('./params.yaml')
    print('PREDICT CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    root = f"./data/{cfg.task.task}/{cfg.split.split}"
    basepath = f"./out/{cfg.task.task}/{cfg.split.split}"
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    ckpt_path = f"{basepath}/{mdir}/best.pt"

    # determine best val split (model) and validate results
    with open(f"{basepath}/{mdir}/metrics.json", 'r') as f:
        metrics = json.load(f)

    # valid_score = metrics.pop('valid')
    # test_score = metrics.pop('test')
    best_fold = np.argmin([v['val_mae'] for k, v in metrics.items()
                           if k not in ['valid', 'test', 'best_fold']])
    print('best', best_fold, 'all', metrics)

    with open(f"{root}/valid{best_fold}.pkl", 'rb') as f:
        valid = pickle.load(f)
    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)
    if cfg.model.model == 'ecfp':
        valid = ECFPDataSplit(valid, nbits=cfg.model.nbits)
        test = ECFPDataSplit(test, nbits=cfg.model.nbits)
    test_loader = DataLoader(test, batch_size=cfg.model.n_batch,
                             shuffle=False, num_workers=8)
    valid_loader = DataLoader(valid, batch_size=cfg.model.n_batch,
                              shuffle=False, num_workers=8)

    if 'mmb' in cfg.model.model:
        head = cfg.head.head
        if cfg.model.model in ['mmb', 'mmb-ft']:
            model = AqueousRegModel(head=head,
                                    finetune=cfg.model.finetune)
            model.head.load_state_dict(torch.load(ckpt_path))
            model.explainer = MolecularSelfAttentionViz(save_heatmap=False)
        elif cfg.model.model in ['mmb-avg', 'mmb-ft-avg']:
            model = BaselineAqueousModel(head=head,
                                         finetune=cfg.model.finetune)
            model.head.load_state_dict(torch.load(ckpt_path))
        elif cfg.model.model == 'ecfp':
            model = ECFPLinear(head=cfg.head.head, dim=cfg.model.nbits)
            model.head.load_state_dict(torch.load(ckpt_path))

        if cfg.model.finetune or 'ft' in cfg.model.model:
            mmb_path = f"{basepath}/{mdir}/best_mmb.pt"
            model.mmb.load_state_dict(torch.load(mmb_path))
            # model.head.load_state_dict(torch.load(ckpt_path))
            # model.explainer = MolecularSelfAttentionViz(save_heatmap=False)

        trainer = pl.Trainer(
            accelerator='gpu',
            gpus=1,
            precision=16,
        )

        metrics[f'val_{best_fold}'] = trainer.validate(model, valid_loader)[0]
        metrics[f'test_{best_fold}'] = trainer.validate(model, test_loader)[0]
        print('val scores', 'best fold: ', metrics[f'val_{best_fold}'],
              'valid', metrics['valid'])
        print('test scores', 'best fold', metrics[f'test_{best_fold}'],
              'valid', metrics['test'])

        all_valid = trainer.predict(model, valid_loader)
        all_test = trainer.predict(model, test_loader)

    elif 'ecfp' in cfg.model.model and cfg.head.head in ['svr', 'rf']:
        with open(ckpt_path, 'rb') as file:
            model = pickle.load(file)
        all_valid = {'preds': model.predict(valid.ecfp),
                     'labels': valid.labels}
        all_test = {'preds': model.predict(test.ecfp),
                    'labels': test.labels}

    results = pd.DataFrame(columns=[
        'SMILES', 'Tokens', 'Prediction', 'Label', 'Split']
    )
    for split, all in list(zip(['test', 'valid'], [all_test, all_valid])):
        # reverse order for consistency with plotting
        if 'ecfp' in cfg.model.model:
            smiles = test.smiles if split == 'test' else valid.smiles
            tokens = None
        elif 'mmb' in cfg.model.model:
            smiles = list(chain(*[f.get('smiles') for f in all]))
            tokens = list(chain(*[f.get('tokens') for f in all]))

        if cfg.head.head in ['svr', 'rf']:
            preds = all.get('preds')
            labels = all.get('labels')
        elif cfg.head.head in ['lin', 'hier']:
            preds = torch.concat([f.get('preds') for f in all]).cpu().numpy()
            labels = torch.concat([f.get('labels') for f in all]).cpu().numpy()

        res = pd.DataFrame({
            'SMILES': smiles,
            'Tokens': tokens,
            'Prediction': preds,
            'Label': labels,
            'Split': split
            })
        results = pd.concat([results, res], axis=0)

    # reset index to correspond to visualization UID
    results = results.reset_index(drop=True)
    results = results.reset_index().rename(columns={'index': 'uid'})
    results.to_csv(f"{basepath}/{mdir}/predictions.csv", index=False)

    ###################################
    # yhat = torch.concat([f.get('preds') for f in all_test])
    # y = torch.concat([f.get('labels') for f in all_test])
    if cfg.head.head in ['svr', 'rf']:
        yhat = torch.tensor(all.get('preds'))
        y = torch.tensor(all.get('labels'))
    elif cfg.head.head in ['lin', 'hier']:
        yhat = torch.concat([f.get('preds') for f in all]).cpu().numpy()
        y = torch.concat([f.get('labels') for f in all]).cpu().numpy()

    if cfg.split.scale:
        scaler = RobustScaler(quantile_range=[10, 90])
        scaler.center_ = -2.68
        scaler.scale_ = 5.779  # 5.8

        yhat = scaler.inverse_transform(torch.reshape(yhat, (1, -1)))
        y = scaler.inverse_transform(torch.reshape(y, (1, -1)))
        # yhat = scaler.inverse_transform(torch.unsqueeze(yhat, 0))
        # y = scaler.inverse_transform(torch.unsqueeze(y, 0))

        yhat = torch.squeeze(torch.tensor(yhat))
        y = torch.squeeze(torch.tensor(y))

    mse = nn.MSELoss()(yhat, y)
    mae = nn.L1Loss()(yhat, y)
    rmse = torch.sqrt(mse)

    # data = pd.DataFrame({'y': y, 'yhat': yhat})
    # reg = LinearRegression()
    # reg.fit(yhat.reshape(-1, 1), y)
    # slo = f"{reg.coef_[0]:.3f}"
#
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
    sns.regplot(x=lim, y=lim, ax=p.ax_joint,
                color='grey', ci=None, scatter=False)
    p.fig.suptitle(f"{cfg.task.plot_title} parity plot \
        \n{_acc} {split}test set")
    p.set_axis_labels(f"Experimental {cfg.task.plot_propname}",
                      f"Model {cfg.task.plot_propname}")

    p.fig.subplots_adjust(top=0.95)
    p.fig.tight_layout()
    txt = f"RMSE = {rmse:.3f}\nMAE = {mae:.3f}\nn = {len(y)}"
    plt.text(lim[1], lim[0], txt, fontsize=14, ha="right", va="bottom")
    # plt.text(1, 0, txt, ha="right", va="bottom", fontsize=14)
    p.savefig(f"{basepath}/{mdir}/parity_plot.png")


if __name__ == "__main__":
    predict_model()
