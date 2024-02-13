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


# def load_metrics(task, split, models):
#     " Load MAE from metrics.json files for each models for given task/split. "
#     basepath = f'/workspace/final/{task}/{split}'
#     data = {}
#
#     for mdir in models:
#         # Load the metrics.json file
#         try:
#             with open(f"{basepath}/{mdir}/metrics.json", 'r') as f:
#                 metrics = json.load(f)
#                 val_mae = [metrics[str(i)]['val_mae'] for i in range(cfg.split.n_splits)]
#                 test_mae = metrics['test']['test_mae']
#                 data[mdir] = {'val_mae': val_mae, 'test_mae': test_mae}
#         except FileNotFoundError:
#             print(f"File not found: {basepath}/{mdir}")
#             continue
#
#     return data


# @hydra.main(
#     version_base="1.3", config_path="../conf", config_name="config")
# def plot_models(cfg: DictConfig) -> None:
def plot_models():
    cfg = OmegaConf.load('./params.yaml')
    print('PLOT MODELS CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    basepath = f"/workspace/final/{cfg.task.task}/{cfg.split.split}"
    models = ['mmb-hier', 'mmb-lin',
              'mmb-ft-hier', 'mmb-ft-lin',
              'mmb-avg-hier', 'mmb-avg-lin',
              'ecfp-hier', 'ecfp-lin',
              'mmb-ft-avg-hier', 'mmb-ft-avg-lin']

    metrics = {}
    for mdir in models:
        try:
            with open(f"{basepath}/{mdir}/metrics.json", 'r') as f:
                metric = json.load(f)
                val_mae = [metric[str(i)]['val_mae'] for i in range(cfg.split.n_splits)]
                test_mae = metric['test']['test_mae']
                metrics[mdir] = {'val_mae': val_mae, 'test_mae': test_mae}
        except FileNotFoundError:
            print(f"File not found: {basepath}/{mdir}")
            continue

    models = metrics.keys()

    # Updated plotting with the revised color scheme
    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.35  # Width of the bars

    colors_dark = plt.cm.viridis(np.linspace(0, 0.75, len(models)))
    colors_light = plt.cm.viridis(np.linspace(0.05, 0.8, len(models)))

    # colors_light = colors_dark + np.array([0.3, 0.3, 0.3, 0])
    # colors_light = np.clip(colors_light, 0, 1)  # Ensuring valid color values

    # base_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    # colors_light = base_colors + np.array([0.3, 0.3, 0.3, 0])
    # colors_light = np.clip(colors_light, 0, 1)  # Ensuring valid color values
    # colors_dark = base_colors

    for i, model in enumerate(models):
        # Placeholder data for CV MAE and Test MAE

        cv_mae_mean = np.mean(metrics[model]['val_mae'])
        cv_mae_std = np.std(metrics[model]['val_mae'])
        test_mae = metrics[model]['test_mae']

        # Plotting CV MAE (lighter color)
        ax.bar(i - width/2, cv_mae_mean, width, yerr=cv_mae_std,
               color=colors_light[i], label=f'{model} (Valid)')

        # Plotting Test MAE (darker color)
        ax.bar(i + width/2, test_mae, width, color=colors_dark[i],
               alpha=0.7, label=f'{model} (Test)')

    # Adding labels and title
    ax.set_ylabel('MAE')
    ax.set_title(f'Model comparison: {cfg.task.plot_title}, {cfg.split.split} \
                 Split, Mean Absolute Error (MAE))')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{basepath}/model_comparision_mae.png")


if __name__ == "__main__":
    plot_models()
    # plot_models(task='aq', split='random')
    # plot_models(task='aq', split='accurate')
    # plot_models(task='aq', split='scaffold')
