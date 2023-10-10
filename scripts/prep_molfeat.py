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
from molfeat.trans.pretrained.dgl_pretrained import PretrainedDGLTransformer
from autosklearn.regression import AutoSklearnRegressor
from molfeat.store import ModelStore
import sklearn
import molfeat
import datamol as dm
# https://molfeat-docs.datamol.io/stable/usage.html#quick-api-tour

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

# smiles, y_true = AqueousDataMolSet()
smiles, y_true = load_dataset(
    "/workspace/data/AqueousSolu.csv", "smiles solute", "logS_aq_avg"
)
smiles = np.array([preprocess_smiles(smi) for smi in smiles])
smiles = np.array([smi for smi in smiles if dm.to_mol(smi) is not None])

print('loaded featurizers')
# Setup the featurizers
trans_ecfp = FPVecTransformer(kind="ecfp:4", length = 512,
                              n_jobs=-1, dtype=np.float32)
trans_ecfp6 = FPVecTransformer(kind="ecfp:6", length = 512,
                               n_jobs=-1, dtype=np.float32)
trans_mordred = FPVecTransformer(kind="mordred", replace_nan=True, n_jobs=-1,
                                 dtype=np.float32)
#trans_chemberta = PretrainedHFTransformer(kind='ChemBERTa-77M-MLM',
                                          #notation='smiles', dtype=np.float32)
#trans_chemberta_mtr = PretrainedHFTransformer(kind='ChemBERTa-77M-MTR',
                                          #notation='smiles', dtype=np.float32)
# trans_jtvae = PretrainedDGLTransformer(kind="jtvae_zinc_no_kl",
#                                       notation='smiles')

feats_ecfp, ind_ecfp = trans_ecfp(smiles, ignore_errors=True)
print(feats_ecfp.shape)
#feats_ecfp6, ind_ecfp6 = trans_ecfp6(smiles, ignore_errors=True)
#print(feats_ecfp6.shape)
#feats_mordred, ind_mordred = trans_mordred(smiles, ignore_errors=True)
#print(feats_mordred.shape)
#feats_chemberta, ind_chemberta = trans_chemberta(smiles, ignore_errors=True)
#print(feats_chemberta.shape)
#feats_jtvae, ind_jtvae = trans_jtvae(smiles, ignore_errors=True)
#print(feats_jtvae.shape)

reps = {
    "smiles": smiles,
    "target": y_true,
    "ecfp": feats_ecfp[ind_ecfp],
    #"ecfp6": feats_ecfp6[ind_ecfp6],
    #"mordred": feats_mordred[ind_mordred],
    #"jtvae": feats_jtvae[ind_jtvae],
    #"chemberta": feats_chemberta[ind_chemberta],
}
print(reps)

with open('/workspace/data/prep/aqueous_prep.pickle', 'wb') as f:
    pickle.dump(reps, f, protocol=pickle.HIGHEST_PROTOCOL)

train_ind, test_ind = scaffold_split(smiles)
print(smiles[test_ind][:5])
