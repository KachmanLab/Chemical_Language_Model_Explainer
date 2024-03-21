
import torch
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import hydra
import json
from omegaconf import OmegaConf, DictConfig
import numpy as np
import seaborn as sns
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

def restore_array(st):
    st = st.strip("'[").strip("']").split(' ')
    st = [s for s in st if s not in ['']]
    return np.array(st, dtype=float)


def plot_similarity():
    cfg = OmegaConf.load('./params.yaml')
    print('SIMILARITY EVALUATION CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    basepath = f"/workspace/final/{cfg.task.task}/{cfg.split.split}"
    models = ['mmb-hier', 'mmb-lin',
              'mmb-ft-hier', 'mmb-ft-lin',
              'mmb-avg-hier', 'mmb-avg-lin',
              'mmb-ft-avg-hier', 'mmb-ft-avg-lin',
              #'ecfp-hier',
              'ecfp-lin', 'ecfp-lin-scaled'
              # 'mmb-ft-lin-scaled', 'mmb-ft-lin-scaled-clamp',
              ]

    attributions = {}
    for mdir in models:
        try:
            df = pd.read_csv(f"{basepath}/{mdir}/attributions.csv")
            attribs = df['atom_weights'].apply(restore_array)
            attributions[mdir] = attribs
        except FileNotFoundError:
            print(f"File not found: {basepath}/{mdir}")
            continue

    attributions['all-equal'] = [
        np.ones_like(attr) / len(attr) for attr in attribs
    ]

    # sanity checking
    # for ix in list(zip(*attributions.values())):
    #     print([len(i) for i in ix])
    #     print(len([i for i in ix]))
    #     print(ix)
    #     print(np.array(ix))
    #     print(np.array([len(i) for i in ix]))
    #     print([np.array(i) for i in ix])
    #     print(np.array([i for i in ix]))

    # similarities = [
    #     cosine_similarity(np.array(ix)) for ix in
    #     list(zip(*attributions.values()))
    # ]

    similarities = []
    mismatch = 0
    for ix in list(zip(*attributions.values())):
        try:
            similarities.append(
                cosine_similarity(np.array(ix))
            )
        except:
            print([len(i) for i in ix])
            mismatch += 1
            # continue
    print('total', len(attributions['all-equal']))
    print('mismatched:', mismatch)
    # similarities = []
    # attribszip = list(zip(*attributions.values()))
    # for id, ix in enumerate(attribszip):
    #     # print(ix)
    #     # print(id)
    #     tmp = [len(i) for i in ix]
    #     if not np.all(tmp[0] == tmp):
    #         print(tmp)
    #     # print(ix)
    #     # print(len(cosine_similarity(np.array(ix))))
    #
    #     # similarities.append(
    #     #     cosine_similarity(np.array(ix))
    #     # )
    # print(similarities)

    similarity_matrix = np.mean(np.array(similarities), axis=0)

    print(len(similarities))
    print(np.array(similarities))
    print(np.array(similarities).shape)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

    sns.set(font_scale=1.1)
    plt.figure(figsize=(10, 8))  # was (10,8)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    sns.heatmap(similarity_matrix,
                annot=True,
                mask=mask,
                xticklabels=attributions.keys(),
                yticklabels=attributions.keys(),
                cbar_kws={'label': 'Average cosine similarity'},
                )

    # plt.title(f"Pairwise cosine similarity of attributed relevance\n\
    #           averaged over {cfg.split.split} test set")

    plt.tight_layout()
    plt.savefig(f"{basepath}/attribution_comparison_{cfg.split.split}.png")


if __name__ == "__main__":
    plot_similarity()
