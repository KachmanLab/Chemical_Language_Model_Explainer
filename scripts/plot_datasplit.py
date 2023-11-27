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
import seaborn as sns
from sklearn.manifold import TSNE
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
    if cfg.model.model == 'mmb':
        # AqueousRegModel(head=head,
        model = MMB_R_Featurizer(head=head,
                                 finetune=cfg.model.finetune)
        model.head.load_state_dict(torch.load(ckpt_path))
    if cfg.model.finetune or cfg.model.model == 'mmb-ft':
        model = MMB_R_Featurizer(head=head,
                                 finetune=cfg.model.finetune)
        # model = model.load_from_checkpoint(ckpt_path, head=head)
        mmb_path = f"{basepath}/{mdir}/best_mmb.pt"
        model.mmb.load_state_dict(torch.load(mmb_path))
        model.head.load_state_dict(torch.load(ckpt_path))
    elif cfg.model.model == 'mmb-avg':
        model = MMB_AVG_Featurizer(head=head,
                                   finetune=cfg.model.finetune)

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

    tsne = TSNE(n_components=2, random_state=42)
    valid_latent = tsne.fit_transform(valid_emb)
    test_latent = tsne.fit_transform(test_emb)

    # Plotting the latent space
    plt.figure(figsize=(10, 8))
    plt.scatter(valid_latent[:, 0], valid_latent[:, 1], c='blue', label='Valid')
    plt.scatter(test_latent[:, 0], test_latent[:, 1], c='red', label='Test')
    plt.xlabel('t-SNE Latent Dimension 1')
    plt.ylabel('t-SNE Latent Dimension 2')
    plt.title(f't-SNE Latent Space Visualization, Valid+Test set\n \
         {cfg.model.model} on {cfg.task.plot_title} {cfg.split.split} split')
    plt.legend()

    # Save or show the plot
    plt.tight_layout()
    plt.savefig(f"{basepath}/{mdir}/latent_viz.png")
    # plt.show()
    # path}/random_df.csv'


if __name__ == "__main__":
    plot_datasplit()
    # plot_datasplit(task='aq', split='random')
    # plot_datasplit(task='aq', split='accurate')
    # plot_datasplit(task='aq', split='scaffold')
