
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
              'ecfp-hier', 'ecfp-lin',
              'mmb-ft-avg-hier', 'mmb-ft-avg-lin']

    attributions = {}
    for mdir in models:
        try:
            df = pd.read_csv(f"{basepath}/{mdir}/attributions.csv")
            attribs = df['atom_weights'].apply(restore_array)
            attributions[mdir] = attribs
        except FileNotFoundError:
            print(f"File not found: {basepath}/{mdir}")
            continue

    attributions['all-equal'] = [np.ones_like(attr) / len(attr) for attr in attribs]

    similarities = []
    # Calculate pairwise cosine similarity
    attribszip = list(zip(*attributions.values()))
    # data = attributions.values()
    # for ix in range(len(attribs)):
    for ix in attribszip:
        print(ix)
        print([len(i) for i in ix])
        similarities.append(
            cosine_similarity(np.array(ix))
        )

    print(similarities)
    similarity_matrix = np.mean(np.array(similarities), axis=0)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix,
                annot=True,
                mask=mask,
                xticklabels=attributions.keys(),
                yticklabels=attributions.keys(),
                )

    plt.title("Pairwise cosine similarity of attributed relevance")
    plt.tight_layout()
    plt.savefig(f"{basepath}/attribution_comparison.png")

    # models = metrics.keys()
    #
    # # Updated plotting with the revised color scheme
    # fig, ax = plt.subplots(figsize=(14, 8))
    # width = 0.35  # Width of the bars
    #
    # colors_dark = plt.cm.viridis(np.linspace(0, 0.75, len(models)))
    # colors_light = plt.cm.viridis(np.linspace(0.05, 0.8, len(models)))
    #
    # # colors_light = colors_dark + np.array([0.3, 0.3, 0.3, 0])
    # # colors_light = np.clip(colors_light, 0, 1)  # Ensuring valid color values
    #
    # # base_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    # # colors_light = base_colors + np.array([0.3, 0.3, 0.3, 0])
    # # colors_light = np.clip(colors_light, 0, 1)  # Ensuring valid color values
    # # colors_dark = base_colors
    #
    # for i, model in enumerate(models):
    #     # Placeholder data for CV MAE and Test MAE
    #
    #     cv_mae_mean = np.mean(metrics[model]['val_mae'])
    #     cv_mae_std = np.std(metrics[model]['val_mae'])
    #     test_mae = metrics[model]['test_mae']
    #
    #     # Plotting CV MAE (lighter color)
    #     ax.bar(i - width/2, cv_mae_mean, width, yerr=cv_mae_std,
    #            color=colors_light[i], label=f'{model} (Valid)')
    #
    #     # Plotting Test MAE (darker color)
    #     ax.bar(i + width/2, test_mae, width, color=colors_dark[i],
    #            alpha=0.7, label=f'{model} (Test)')
    #
    # # Adding labels and title
    # ax.set_ylabel('MAE')
    # ax.set_title(f'Model comparison: {cfg.task.plot_title}, {cfg.split.split} \
    #              Split, Mean Absolute Error (MAE))')
    # ax.set_xticks(range(len(models)))
    # ax.set_xticklabels(models, rotation=45, ha='right')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #
    # plt.tight_layout()
    # plt.savefig(f"{basepath}/model_comparision_mae.png")


if __name__ == "__main__":
    plot_similarity()
    # plot_models(task='aq', split='random')
    # plot_models(task='aq', split='accurate')
    # plot_models(task='aq', split='scaffold')
