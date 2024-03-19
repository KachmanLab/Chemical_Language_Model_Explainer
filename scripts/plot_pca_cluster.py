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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from src.dataloader import ECFPDataSplit
from src.model import MMB_R_Featurizer, MMB_AVG_Featurizer
from src.dataloader import ECFPDataSplit
import os 
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from io import BytesIO

@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def plot_pca_cluster(cfg: DictConfig) -> None:

    cfg = OmegaConf.load('./params.yaml')
    print('CLUSTER CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    basepath = f'./final/{cfg.task.task}/{cfg.split.split}'
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    ckpt_path = f"{basepath}/{mdir}/best.pt"

    with open(f"{basepath}/{mdir}/metrics.json", 'r') as f:
        metrics = json.load(f)
        best_fold = metrics['best_fold']
    # best_fold = '1'

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
    # else:
    #     raise NotImplementedError

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


    pca = PCA(n_components=2, random_state=cfg.split.data_seed)
    pca = pca.fit(valid_emb)
    valid_latent = pca.transform(valid_emb)
    test_latent = pca.transform(test_emb)

    valid_label = np.array(valid.labels)
    test_label = np.array(test.labels)

    # all_latent = np.concatenate([valid_latent, test_latent])
    n_neighbors = 6
    n_clusters = 16

    # kmeans = KMeans(n_clusters, random_state=cfg.split.data_seed)
    # kmeans = kmeans.fit(valid_latent)
    # kmeans = kmeans.fit(test_latent)
    # print(kmeans.predict(test_latent))
    # kdist = kmeans.transform(test_latent)

    kmeans = KMeans(n_clusters, random_state=cfg.split.data_seed)
    kmeans = kmeans.fit(test_emb)
    kdist = kmeans.transform(test_emb)

    print('kd shape', kdist.shape)
    print(kdist)
    neighbor_ix = [np.argsort(kdist[:, cl])[:n_neighbors] for cl in range(n_clusters)]

    neighbors = np.concatenate(neighbor_ix)
    print(len(neighbor_ix), neighbor_ix)

    cmap = plt.cm.viridis
    expl_var = pca.explained_variance_ratio_ * 100
    print('expl.var', sum(expl_var), expl_var)
    plt.figure(figsize=(10, 8))

    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=269,
        linewidths=3,
        color="b",
        zorder=10,
        label='Centroids'
    )
    if n_clusters < 5:
        plt.scatter(
            test_latent[neighbors, 0],
            test_latent[neighbors, 1],
            marker="x",
            s=69,
            linewidths=3,
            color="black",
            zorder=10,
            label='Neighbours'
        )

    val_sc = plt.scatter(valid_latent[:, 0], valid_latent[:, 1],
                         c=valid_label, cmap=cmap, marker='o', label='Valid')
    test_sc = plt.scatter(test_latent[:, 0], test_latent[:, 1],
                          c=test_label, cmap=cmap, marker='^', label='Test')
    plt.xlabel(f'PCA Dimension 1 ({expl_var[0]:.2f}% explained variance)',
               fontsize=18)
    plt.ylabel(f'PCA Dimension 2 ({expl_var[1]:.2f}% explained variance)',
               fontsize=18)
    # plt.title(f'PCA Latent Space Visualization of {cfg.model.model}-{cfg.head.head}\
    #           \n{cfg.task.plot_title}, {cfg.split.split} split')
    plt.legend(fontsize=16)

    cbar = plt.colorbar(val_sc, orientation='vertical')
    cbar.set_label(f'{cfg.task.plot_propname}',
                   fontsize=16)

    # Save or show the plot
    plt.tight_layout()
    plt.savefig(f"{basepath}/{mdir}/pca_kmeans_viz.png")




    # PLOT 1 grid of mol's 
    # fig, axs = plt.subplots(n_neighbors, n_clusters,
    #                         figsize=(n_clusters*9, n_neighbors*8))
    # for i, cluster in enumerate(neighbor_ix[:n_clusters]):
    #     for j, uid in enumerate(cluster[:n_neighbors]):
    #         fig = Image.open(f"{basepath}/{mdir}/viz/{uid}_{mdir}_MolViz.png")
    #         # axs[i, j].imshow(molfig[i*n_cols+j])
    #         print(i, j, '\t', cluster, uid)
    #         axs[j, i].imshow(fig)
    #         axs[j, i].axis('off')
    #
    # plt.tight_layout()
    # plt.savefig(f"{basepath}/{mdir}/cluster_grid_viz.png")



    # models = ['mmb-ft-lin', 'mmb-hier', 'mmb-ft-avg-hier', 'mmb-avg-hier', 'ecfp-lin']
    # models = ['mmb-ft-lin', 'mmb-ft-avg-hier', 'ecfp-lin']
    models = ['mmb-ft-lin', 'mmb-ft-avg-lin', 'ecfp-lin-scaled']
    n_models = len(models)
    # loop over clusters, one fig per cluster
    for p, cluster in enumerate(neighbor_ix[:n_clusters]):
        fig, axs = plt.subplots(n_neighbors, n_models,
                                figsize=(n_models*9, n_neighbors*8))
        for i, mdir in enumerate(models):
            if 'ecfp' in mdir:
                fname = 'MorganAttrib_reg'
            else:
                mname = mdir.replace('-lin', '').replace('-hier', '')
                fname = f"{mname}_MolViz"

            for j, uid in enumerate(cluster[:n_neighbors]):
                fig = Image.open(f"{basepath}/{mdir}/viz/{uid}_{fname}.png")
                # axs[i, j].imshow(molfig[i*n_cols+j])
                print(i, j, '\t', cluster, uid)
                axs[j, i].imshow(fig)
                axs[j, i].axis('off')

        # plt.title("Cluster {p}, KMeans, {models}")
        plt.tight_layout()
        plt.savefig(f"{basepath}/cluster/cluster{p}_grid_viz.png")

if __name__ == "__main__":
    plot_pca_cluster()
