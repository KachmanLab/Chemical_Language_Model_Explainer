import torch
import json
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np
# import seaborn as sns
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.dataloader import ECFPDataSplit
from src.model import MMB_R_Featurizer, MMB_AVG_Featurizer


@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def plot_datasplit(cfg: DictConfig) -> None:

    cfg = OmegaConf.load('./params.yaml')
    print('PREDICT CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    basepath = f'./out/{cfg.task.task}/{cfg.split.split}'
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    ckpt_path = f"{basepath}/{mdir}/best.pt"

    with open(f"{basepath}/{mdir}/metrics.json", 'r') as f:
        metrics = json.load(f)
        best_fold = metrics['best_fold']

    root = f"./data/{cfg.task.task}/{cfg.split.split}"
    with open(f"{root}/valid{best_fold}.pkl", 'rb') as f:
        valid = pickle.load(f)
    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)
    test_loader = DataLoader(test, batch_size=cfg.model.n_batch,
                             shuffle=False, num_workers=8)
    valid_loader = DataLoader(valid, batch_size=cfg.model.n_batch,
                              shuffle=False, num_workers=8)

    head = cfg.head.head
    if cfg.model.model in ['mmb', 'mmb-ft']:
        # AqueousRegModel(head=head,
        model = MMB_R_Featurizer(head=head,
                                 finetune=cfg.model.finetune)
        model.head.load_state_dict(torch.load(ckpt_path))
    elif cfg.model.model in ['mmb-avg', 'mmb-ft-avg']:
        model = MMB_AVG_Featurizer(head=head,
                                   finetune=cfg.model.finetune)
    elif cfg.model.model == 'ecfp':
        valid_emb = np.array(ECFPDataSplit(valid).ecfp)
        test_emb = np.array(ECFPDataSplit(test).ecfp)

    if cfg.model.finetune or 'ft' in cfg.model.model:
        mmb_path = f"{basepath}/{mdir}/best_mmb.pt"
        model.mmb.load_state_dict(torch.load(mmb_path))
        model.head.load_state_dict(torch.load(ckpt_path))

    if 'mmb' in cfg.model.model:
        trainer = pl.Trainer(
            accelerator='gpu',
            gpus=1,
            precision=16,
        )
        valid_emb = trainer.predict(model, valid_loader)
        test_emb = trainer.predict(model, test_loader)
        valid_emb = np.array(torch.concat(valid_emb))
        test_emb = np.array(torch.concat(test_emb))

    # for i in range(len(valid_emb)):
    #     print(np.array(valid_emb[i]).shape, np.array(test_emb[i]).shape)

    # tsne = TSNE(n_components=2, random_state=42)
    # tsne = tsne.fit(valid_emb)
    # valid_latent = tsne.transform(valid_emb)
    # test_latent = tsne.transform(test_emb)

    ##########################

    weights = model.head.fc1.weight[0].cpu().detach().numpy()
    bias = model.head.fc1.bias.cpu().detach().numpy()
    # activations = test_emb
    activations = valid_emb
    positive_count = np.array([(act > 0.05).sum() for act in activations]).mean()
    negative_count = np.array([(act < 0.05).sum() for act in activations]).mean()
    zero_count = (activations == 0).sum()

    print('raw emb')
    print('pos', positive_count)
    print('neg', negative_count)
    print('zero', zero_count)

    print(valid_emb.shape, weights.shape)
    ds_mean = np.array(valid.labels).mean()
    print('val mean', ds_mean, 'bias', bias)

    if cfg.head.head == 'lin':
        activations = valid_emb @ weights + bias  # - ds_mean
        positive_count = np.array([(act > 0.05).sum() for act in activations]).mean()
        negative_count = np.array([(act < 0.05).sum() for act in activations]).mean()
        zero_count = (activations == 0).sum()

        print('activations')
        print('pos', positive_count)
        print('neg', negative_count)
        print('zero', zero_count)

    ##########################

    pca = PCA(n_components=2, random_state=cfg.split.data_seed)
    pca = pca.fit(valid_emb)
    valid_latent = pca.transform(valid_emb)
    test_latent = pca.transform(test_emb)

    valid_label = np.array(valid.labels)
    test_label = np.array(test.labels)
    cmap = plt.cm.viridis

    # Plotting the latent space
    plt.figure(figsize=(10, 8))
    # plt.scatter(valid_latent[:, 0], valid_latent[:, 1], c='blue', label='Valid')
    # plt.scatter(test_latent[:, 0], test_latent[:, 1], c='red', label='Test')

    val_sc = plt.scatter(valid_latent[:, 0], valid_latent[:, 1],
                         c=valid_label, cmap=cmap, marker='o', label='Valid')
    test_sc = plt.scatter(test_latent[:, 0], test_latent[:, 1],
                          c=test_label, cmap=cmap, marker='^', label='Test')
    plt.xlabel('PCA Latent Dimension 1')
    plt.ylabel('PCA Latent Dimension 2')
    plt.title(f'PCA Latent Space Visualization of {cfg.model.model}-{cfg.head.head}\
              \n{cfg.task.plot_title}, {cfg.split.split} split')
    plt.legend()

    cbar = plt.colorbar(val_sc, orientation='vertical')
    cbar.set_label(f'{cfg.task.plot_propname}')

    # Save or show the plot
    plt.tight_layout()
    plt.savefig(f"{basepath}/{mdir}/latent_viz.png")
    # plt.show()
    # path}/random_df.csv'


if __name__ == "__main__":
    # params = {'axes.labelsize': 16,
    #           'axes.titlesize': 16}
    # plt.rcParams.update(params)
    plot_datasplit()
    # plot_datasplit(task='aq', split='random')
    # plot_datasplit(task='aq', split='accurate')
    # plot_datasplit(task='aq', split='scaffold')
