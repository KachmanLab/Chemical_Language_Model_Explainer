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
import numpy as np
import seaborn as sns

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
    best_fold = np.argmin([v['val_mae'] for k, v in metrics.items() if k not in ['valid', 'test']])
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

    head = cfg.head.head
    if cfg.model.model == 'mmb':
        model = AqueousRegModel(head=head,
                                finetune=cfg.model.finetune)
        model.head.load_state_dict(torch.load(ckpt_path))
    if cfg.model.finetune or cfg.model.model == 'mmb-ft':
        model = AqueousRegModel(head=head,
                                finetune=cfg.model.finetune)
        # model = model.load_from_checkpoint(ckpt_path, head=head)
        mmb_path = f"{basepath}/{mdir}/best_mmb.pt"
        model.mmb.load_state_dict(torch.load(mmb_path))
        model.head.load_state_dict(torch.load(ckpt_path))
    elif cfg.model.model == 'mmb-avg':
        model = BaselineAqueousModel(head=head,
                                     finetune=cfg.model.finetune)
    elif cfg.model.model == 'ecfp':
        model = ECFPLinear(head=cfg.head.head, dim=cfg.model.nbits)
        model.head.load_state_dict(torch.load(ckpt_path))

    trainer = pl.Trainer(
        accelerator='gpu',
        gpus=1,
        precision=16,
    )

    metrics[f'val_{best_fold}'] = trainer.validate(model, valid_loader)[0]
    metrics[f'test_{best_fold}'] = trainer.validate(model, test_loader)[0]
    print('val scores', metrics[f'val_{best_fold}'], metrics['valid'])
    print('test scores', metrics[f'test_{best_fold}'], metrics['test'])

    all_valid = trainer.predict(model, valid_loader)
    all_test = trainer.predict(model, test_loader)

    results = pd.DataFrame(columns=[
        'SMILES', 'Tokens', 'Prediction', 'Label', 'Split']
    )
    for split, all in list(zip(['test', 'valid'], [all_test, all_valid])):
        # reverse order for consistency with plotting
        if cfg.model.model == 'ecfp':
            smiles = test.smiles if split == 'test' else valid.smiles
            tokens = None
        else:
            smiles = list(chain(*[f.get('smiles') for f in all]))
            tokens = list(chain(*[f.get('tokens') for f in all]))
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
    yhat = torch.concat([f.get('preds') for f in all_test])
    y = torch.concat([f.get('labels') for f in all_test])

    mse = nn.MSELoss()(yhat, y)
    mae = nn.L1Loss()(yhat, y)
    rmse = torch.sqrt(mse)

    data = pd.DataFrame({'y': y, 'yhat': yhat})
    reg = LinearRegression()
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
    plt.text(lim[1], lim[0], txt, ha="right", va="bottom", fontsize=14)
    # plt.text(1, 0, txt, ha="right", va="bottom", fontsize=14)
    p.savefig(f"{basepath}/{mdir}/parity_plot.png")


if __name__ == "__main__":
    predict_model()
