import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import dagshub
# from dagshub.pytorch_lightning import DAGsHubLogger
import mlflow
# from src.dataloader import AqSolDataset, AqSolDeepChem
# from src.datamol_loader import * #AqueousDataMolSet, scaffold_split
from src.model import AqueousRegModel, BaselineAqueousModel
import pickle

from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.model_selection import GroupShuffleSplit
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import root_mean_squared_error, mean_absolute_error
import sklearn
import molfeat
import datamol as dm
# https://molfeat-docs.datamol.io/stable/usage.html#quick-api-tour

import warnings
warnings.filterwarnings("ignore", module="autosklearn")

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])


# load preprocessed representations
with open('/workspace/data/prep/aqueous_prep.pickle', 'rb') as f:
#with open('/workspace/data/prep/aqueous_mmb.pickle', 'rb') as f:
    reps = pickle.load(f)

#smiles, y_true = reps.pop(['smiles', 'labels'])
smiles = reps.pop('smiles')
y_true = reps.pop('target')


def scaffold_split(smiles):
    """In line with common practice, we will use the scaffold split to evaluate our models"""
    scaffolds = [dm.to_smiles(dm.to_scaffold_murcko(dm.to_mol(smi))) for smi in smiles]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg['seed'])
    return next(splitter.split(smiles, groups=scaffolds))
train_ind, test_ind = scaffold_split(smiles)

for name, feats in reps.items():
    print(name)
    print(feats.shape)
    print(feats[:2, :8])
    print('***')
# https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_resampling.html#sphx-glr-examples-40-advanced-example-resampling-py
solu_scores = {}
for name, feats in reps.items():
    # Train
    automl = AutoSklearnRegressor(
        memory_limit=24576,
        time_left_for_this_task=180,
        per_run_time_limit=30,
        n_jobs=1,
        metric=[root_mean_squared_error, mean_absolute_error],
        seed=cfg['seed'],
        include={
            'feature_preprocessor':['no_preprocessing'],
            'regressor':['gaussian_process', 'mlp', 'libsvm_svr', 'sgd'],
        },
        #tmp_folder='/workspace/results/aqueous/autosk',
        #resampling_strategy=scaffold_split,
        resampling_strategy='cv',
        resampling_strategy_arguments={"folds": 5},
    )

    print(f"fitting {name}, shape {feats.shape}")
    automl.fit(feats[train_ind], y_true[train_ind])
    #y: array-like, shape = [n_samples] or [n_samples, n_targets]
    #automl.refit(feats[train_ind], y_true[train_ind])
    print(automl.leaderboard())

    print("predicting")
    # Predict and evaluate
    y_hat = automl.predict(feats[test_ind])

    # Evaluate
    mae = mean_absolute_error(y_true[test_ind], y_hat)
    rmse = root_mean_squared_error(y_true[test_ind], y_hat)
    skmae = sk_mae(y_true[test_ind], y_hat)
    skrmse = sk_mse(y_true[test_ind], y_hat, squared=False)
    print(name, 'mae', mae, 'rmse', rmse, 'skmae', skmae, 'skrmse', skrmse)

    solu_scores[name] = {'mae': mae, 'rmse': rmse, 'skmae': str(skmae), 'skrmse': str(skrmse)}
    with open(f"/workspace/results/aqueous/autosk_results{cfg['seed']}.json", 'w') as f:
        json.dump(solu_scores, f)

print(solu_scores)

#{'mmb': {'mae': 0.9973724664613635, 'rmse': 1.2760792284047846}, 'ecfp': {'mae': 1.2944131871049267, 'rmse': 1.672714510743058}, 'ecfp6': {'mae': 1.2521010247613389, 'rmse': 1.6126655041170632}, 'mordred': {'mae': 0.8678289758449074, 'rmse': 1.142530401832952}}

#results = pd.DataFrame(
    #columns=['SMILES', 'Tokens', 'logS_pred', 'logS_exp', 'Atom_weights', 'Split']
#)
#for split, all in list(zip(['test', 'val', 'train'], [test, val, train])):
    ## reverse order for consistency with plotting
    #smiles = list(chain(*[f.get('smiles') for f in all]))
    #tokens = list(chain(*[f.get('tokens') for f in all]))
    #atom_weights = list(chain(*[f.get('atom_weights') for f in all]))
    #preds = torch.concat([f.get('preds') for f in all]).cpu().numpy()
    #labels = torch.concat([f.get('labels') for f in all]).cpu().numpy()
#
    #res = pd.DataFrame({
        #'SMILES': smiles,
        #'Tokens': tokens,
        #'logS_pred': preds,
        #'logS_exp': labels,
        #'Atom_weights': atom_weights,
        #'Split': split
        #})
    #results = pd.concat([results, res], axis=0)

# reset index to correspond to visualization UID
#results = results.reset_index(drop=True)
#results = results.reset_index().rename(columns={'index':'uid'})
#results.to_csv('/workspace/results/aqueous/AqueousSolu_predictions.csv', index=False)
