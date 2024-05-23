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

    models = [
        'mmb-ft-lin', 'mmb-ft-hier',
        'mmb-lin', 'mmb-hier',
        'mmb-ft-avg-lin', 'mmb-ft-avg-hier',
        'mmb-avg-lin', 'mmb-avg-hier',
        'ecfp-lin', 'ecfp2k-lin',
        'ecfp-lin-scaled', 'ecfp2k-lin-scaled',
        'ecfp-hier', 'ecfp2k-hier',
        'ecfp-svr', 'ecfp2k-svr',
        'ecfp-rf', 'ecfp2k-rf',
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
                elif 'val_mse' in metric[str(0)].keys():
                    val_rmse = [np.sqrt(
                        metric[str(i)]['val_mse']) for i in range(cfg.split.n_splits)
                    ]
                    test_rmse = np.sqrt(metric['test']['test_mse'])
                else:
                    val_rmse = [0. for i in range(cfg.split.n_splits)]
                    test_rmse = 0.

                if cfg.split.scale:
                    if cfg.split.split == 'scaffold':
                        # scaler.center_ = -2.61
                        scale_ = 5.657
                    elif cfg.split.split in ['accurate', 'random']:
                        # scaler.center_ = -2.68
                        scale_ = 5.779

                    val_mae = [e * scale_ for e in val_mae]
                    test_mae = test_mae * scale_
                    val_rmse = [e * scale_ for e in val_rmse]
                    test_rmse = test_rmse * scale_

                metrics[mdir] = {'val_mae': val_mae, 'test_mae': test_mae,
                                 'val_rmse': val_rmse, 'test_rmse': test_rmse}

        except FileNotFoundError:
            print(f"File not found: {basepath}/{mdir}")
            continue

    models = metrics.keys()

    colors_dark = plt.cm.viridis(np.linspace(0, 0.75, len(models)))
    colors_light = plt.cm.viridis(np.linspace(0.05, 0.8, len(models)))

    res = []
    for error in ['mae', 'rmse']:
        # Updated plotting with the revised color scheme
        fig, ax = plt.subplots(figsize=(14, 6))
        width = 0.35  # Width of the bars

        for i, model in enumerate(models):
            # Placeholder data for CV MAE and Test MAE
            cv_errs = metrics[model][f'val_{error}']
            test_err = metrics[model][f'test_{error}']

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

            res.append({
                'Model': model,
                f'{error}-mean': np.round(cv_err_mean, 4),
                f'{error}-std': np.round(cv_err_std, 4),
                f'{error}-min': np.round(cv_err_min, 4),
                f'{error}-max': np.round(cv_err_max, 4),
                f'{error}-test': np.round(test_err, 4),
            })
        # Adding labels and title
        if error == 'mae':
            ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=15)
        elif error == 'rmse':
            ax.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=15)
        plt.gca().tick_params(axis='y', labelsize='large')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, ha='center', rotation=75, fontsize=14)
        # ax.set_xticklabels(models, ha='center', fontsize=13)
        # ax.set_xticklabels(models, rotation=45, ha='right')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{basepath}/model_comparison_{error}.png")
        plt.clf()

    df = pd.DataFrame(res)
    df.set_index('Model', inplace=True)
    df = df.pivot_table(index='Model')
    df = df[['mae-mean', 'mae-std', 'mae-min', 'mae-max', 'mae-test',
            'rmse-mean', 'rmse-std', 'rmse-min', 'rmse-max', 'rmse-test']]
    df.to_csv(f"{basepath}/model_metrics.csv")
    print(df)

if __name__ == "__main__":
    plot_models()
    # plot_models(task='aq', split='random')
    # plot_models(task='aq', split='accurate')
    # plot_models(task='aq', split='scaffold')
