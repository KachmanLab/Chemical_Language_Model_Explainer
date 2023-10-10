import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import dagshub
# from dagshub.pytorch_lightning import DAGsHubLogger
import mlflow
# from src.dataloader import AqSolDataset, AqSolDeepChem
from src.datamol_loader import * #AqueousDataMolSet, scaffold_split
from src.model import AqueousRegModel, BaselineAqueousModel


from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from autosklearn.regression import AutoSklearnRegressor

import sklearn
import molfeat
import datamol as dm
# https://molfeat-docs.datamol.io/stable/usage.html#quick-api-tour

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])


# load preprocessed representations
reps = pickle.load('/workspace/data/prep/aqueous_prep.pickle',)

smiles, y_true = reps.pop(['smiles', 'labels'])
# Train a model
train_ind, test_ind = scaffold_split(smiles)

solu_scores = {}
for name, feats in reps.items():
    # Train
    automl = AutoSklearnRegressor(
        memory_limit=24576,
        # For practicality’s sake, limit this to 5 minutes!
        # (x3 = 15 min in total)
        time_left_for_this_task=60,
        per_run_time_limit=20,
        n_jobs=-1,
        seed=cfg['seed'],
    )
    print("fitting")
    automl.fit(feats[train_ind], y_true[train_ind])

    # Predict and evaluate
    y_hat = automl.predict(feats[test_ind])

    # Evaluate
    mae = mean_absolute_error(y_true[test_ind], y_hat)
    rmse = mean_squared_error(y_true[test_ind], y_hat, squared=False)
    solu_scores[name] = {'mae': mae, 'rmse': rmse}

print(solu_scores)

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
