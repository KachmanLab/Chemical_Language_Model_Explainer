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
from sklearn.preprocessing import RobustScaler


def plot_models():
    cfg = OmegaConf.load('./params.yaml')
    print('PLOT MODELS CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    basepath = f"/workspace/final/{cfg.task.task}/{cfg.split.split}"
    models = ['ecfp-rf', 'ecfp-sverad', 'ecfp2k-rf', 'ecfp2k-sverad']

    models = ['mmb-hier', 'mmb-lin',
              'mmb-ft-hier', 'mmb-ft-lin',
              'mmb-avg-hier', 'mmb-avg-lin',
              'mmb-ft-avg-hier', 'mmb-ft-avg-lin',
              'ecfp-hier', 'ecfp-lin',
              'ecfp2k-hier', 'ecfp2k-lin',
              'ecfp-rf', 'ecfp-sverad',
              'ecfp2k-rf', 'ecfp2k-sverad'
              ]

    metrics = {}
    for mdir in models:
        try:
            with open(f"{basepath}/{mdir}/metrics.json", 'r') as f:
                metric = json.load(f)
                print(mdir, metric.keys())
                print('asserting', str(cfg.split.n_splits-1)
                      in list(metric.keys()))
                val_mae = [
                    metric[str(i)]['val_mae'] for i in range(cfg.split.n_splits)
                ]
                test_mae = metric['test']['test_mae']

                if 'val_rmse' in metric[str(0)].keys():
                    val_rmse = [
                        metric[str(i)]['val_rmse'] for i in range(cfg.split.n_splits)
                    ]
                    test_rmse = metric['test']['test_rmse']
                else:
                    val_rmse = [0. for i in range(cfg.split.n_splits)]
                    test_rmse = 0.

                metrics[mdir] = {'val_mae': val_mae, 'test_mae': test_mae,
                                 'val_rmse': val_rmse, 'test_rmse': test_rmse}

        except FileNotFoundError:
            print(f"File not found: {basepath}/{mdir}")
            continue

    models = metrics.keys()

    colors_dark = plt.cm.viridis(np.linspace(0, 0.75, len(models)))
    colors_light = plt.cm.viridis(np.linspace(0.05, 0.8, len(models)))

    # colors_light = colors_dark + np.array([0.3, 0.3, 0.3, 0])
    # colors_light = np.clip(colors_light, 0, 1)  # Ensuring valid color values

    # base_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    # colors_light = base_colors + np.array([0.3, 0.3, 0.3, 0])
    # colors_light = np.clip(colors_light, 0, 1)  # Ensuring valid color values
    # colors_dark = base_colors

    for scale in [False, True]:
        for error in ['mae', 'rmse']:
        # for error in ['mae']:
            # Updated plotting with the revised color scheme
            fig, ax = plt.subplots(figsize=(14, 6))
            width = 0.35  # Width of the bars

            for i, model in enumerate(models):
                # Placeholder data for CV MAE and Test MAE
                cv_errs = metrics[model][f'val_{error}']
                test_err = metrics[model][f'test_{error}']

                if cfg.split.scale and scale:
                    suffix = '_scaled'
                    scaler = RobustScaler(quantile_range=[10, 90])
                    if cfg.split.split == 'scaffold':
                        scaler.center_ = -2.61
                        scaler.scale_ = 5.657
                    elif cfg.split.split in ['accurate', 'random']:
                        scaler.center_ = -2.68
                        scaler.scale_ = 5.779

                    # cv_errs = [scaler.inverse_transform(e) for e in cv_errs
                    cv_errs = scaler.inverse_transform(np.array(
                        cv_errs).reshape(1, -1))[0].tolist()
                    test_err = scaler.inverse_transform(np.array(
                        test_err).reshape(1, -1))[0].tolist()
                else:
                    suffix = ''

                cv_err_mean = np.mean(cv_errs)
                cv_err_std = np.std(cv_errs)

                # plot MAE from 5 CV as points instead of yerr
                for j, cv_err in enumerate(cv_errs):
                    ax.scatter(i - width/2, cv_err, color=colors_light[i],
                               edgecolor='black', zorder=3)

                # Plotting the average CV MAE as a line
                # ax.plot([i - width/2 - 0.1, i - width/2 + 0.1],
                # [cv_mae_mean, cv_mae_mean], color='red', linewidth=2)

                # Plotting CV MAE (lighter color)
                ax.bar(i - width/2, cv_err_mean, width,
                       color=colors_light[i], label=f'{model} (Valid)')

                # Plotting Test MAE (darker color)
                ax.bar(i + width/2, test_err, width,
                       color=colors_dark[i], label=f'{model} (Test)')

                cv_err_min = np.min(metrics[model][f'val_{error}'])
                cv_err_max = np.max(metrics[model][f'val_{error}'])
                print('err', error, model, cv_err_min, cv_err_max)

            # Adding labels and title
            if error == 'mae':
                ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=15)
            elif error == 'rmse':
                ax.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=15)
            plt.gca().tick_params(axis='y', labelsize='large')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, ha='center', rotation=30, fontsize=14)
            # ax.set_xticklabels(models, ha='center', fontsize=13)
            # ax.set_xticklabels(models, rotation=45, ha='right')
            # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(f"{basepath}/model_comparison_{error}{suffix}.png")
            plt.clf()


if __name__ == "__main__":
    plot_models()
    # plot_models(task='aq', split='random')
    # plot_models(task='aq', split='accurate')
    # plot_models(task='aq', split='scaffold')
