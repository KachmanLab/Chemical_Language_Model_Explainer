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


def plot_models():
    cfg = OmegaConf.load('./params.yaml')
    print('PLOT MODELS CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    basepath = f"/workspace/final/{cfg.task.task}/{cfg.split.split}"
    models = ['ecfp-rf', 'ecfp-sverad', 'ecfp2k-rf', 'ecfp2k-sverad']

    # models = ['mmb-hier', 'mmb-lin',
    #           'mmb-ft-hier', 'mmb-ft-lin',
    #           'mmb-avg-hier', 'mmb-avg-lin',
    #           'mmb-ft-avg-hier', 'mmb-ft-avg-lin',
    #           'ecfp-hier', 'ecfp-lin',
    #           'ecfp-rf', 'ecfp-svr',
    #           ]

    metrics = {}
    for mdir in models:
        try:
            with open(f"{basepath}/{mdir}/metrics.json", 'r') as f:
                metric = json.load(f)
                print(mdir, metric.keys())
                print('asserting', str(cfg.split.n_splits-1)
                      in list(metrics.keys()))
                val_mae = [
                    metric[str(i)]['val_mae'] for i in range(cfg.split.n_splits)
                ]
                val_rmse = [
                    metric[str(i)]['val_rmse'] for i in range(cfg.split.n_splits)
                ]
                test_mae = metric['test']['test_mae']
                metrics[mdir] = {'val_mae': val_mae, 'test_mae': test_mae}
        except FileNotFoundError:
            print(f"File not found: {basepath}/{mdir}")
            continue

    models = metrics.keys()

    # Updated plotting with the revised color scheme
    fig, ax = plt.subplots(figsize=(14, 6))
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
        # cv_mae_std = np.std(metrics[model]['val_mae'])
        test_mae = metrics[model]['test_mae']

        # plot MAE from 5 CV as points instead of yerr
        for j, cv_mae in enumerate(cv_maes):
            ax.scatter(i - width/2, cv_mae, color=colors_light[i], edgecolor='black', zorder=3)

        # Plotting the average CV MAE as a line
        # ax.plot([i - width/2 - 0.1, i - width/2 + 0.1], [cv_mae_mean, cv_mae_mean], color='red', linewidth=2)

        # Plotting CV MAE (lighter color)
        ax.bar(i - width/2, cv_mae_mean, width, # yerr=cv_mae_std,
               color=colors_light[i], label=f'{model} (Valid)')

        # Plotting Test MAE (darker color)
        ax.bar(i + width/2, test_mae, width, 
               color=colors_dark[i], label=f'{model} (Test)')

        cv_mae_min = np.min(metrics[model]['val_mae'])
        cv_mae_max = np.max(metrics[model]['val_mae'])
        print('MAE', model, cv_mae_min, cv_mae_max)
        # print(metrics[model]['val_rmse'])
        cv_rmse_min = np.min(metrics[model]['val_rmse'])
        cv_rmse_max = np.max(metrics[model]['val_rmse'])
        print('RMSE', model, cv_rmse_min, cv_rmse_max)

    # Adding labels and title
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=15)
    plt.gca().tick_params(axis='y', labelsize='large')
    # ax.set_title(f'Model comparison: {cfg.task.plot_title}, {cfg.split.split} split')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, ha='center', rotation=30, fontsize=14)
    # ax.set_xticklabels(models, ha='center', fontsize=13)
    # ax.set_xticklabels(models, rotation=45, ha='right')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{basepath}/model_comparison_mae.png")


if __name__ == "__main__":
    plot_models()
    # plot_models(task='aq', split='random')
    # plot_models(task='aq', split='accurate')
    # plot_models(task='aq', split='scaffold')
