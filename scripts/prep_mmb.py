import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import dagshub
# from dagshub.pytorch_lightning import DAGsHubLogger
import mlflow
from src.dataloader import AqSolDataset, AqSolDeepChem
from src.datamol_loader import * #AqueousDataMolSet, scaffold_split
from src.model import AqueousRegModel, BaselineAqueousModel, MMBFeaturizer


from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from molfeat.trans.fp import FPVecTransformer

import sklearn
# import molfeat
# import datamol as dm
# https://molfeat-docs.datamol.io/stable/usage.html#quick-api-tour

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

#smiles, y_true = load_dataset(
    #"/workspace/data/AqueousSolu.csv", "smiles solute", "logS_aq_avg"
#)
#smiles = np.array([preprocess_smiles(smi) for smi in smiles])
#smiles = np.array([smi for smi in smiles if dm.to_mol(smi) is not None])
smiles = all_dataset.smiles
y_true = all_dataset.labels.numpy()

print('loading mmb')
model = MMBFeaturizer()

all_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train', False, 
    1., data_seed=cfg['seed'], augment=False)
all_loader = DataLoader(all_dataset, batch_size=32,
    shuffle=False, num_workers=8)

trainer = pl.Trainer(
    accelerator='gpu',
    gpus=1,
    precision=16,
)

feats_mmb = trainer.predict(model, all_loader)
feats_mmb = torch.cat(feats_mmb).numpy()

# Setup the ECFP featurizers
trans_ecfp = FPVecTransformer(kind="ecfp:4", length = 512,
                              n_jobs=-1, dtype=np.float32)
feats_ecfp, ind_ecfp = trans_ecfp(smiles, ignore_errors=True)

assert feats_mmb.shape == feats_ecfp.shape

reps = {
    "smiles": smiles,
    "target": y_true,
    "mmb": feats_mmb, #[ind_mmb]
    "ecfp": feats_ecfp[ind_ecfp],
}

with open('/workspace/data/prep/aqueous_mmb.pickle', 'wb') as f: 
    pickle.dump(reps, f)

train_ind, test_ind = scaffold_split(smiles)

print(feats_mmb[train_ind].shape, feats_mmb[test_ind].shape)
print(feats_ecfp[train_ind].shape, feats_ecfp[test_ind].shape)








# https://molfeat-docs.datamol.io/stable/api/molfeat.calc.html#molfeat.calc.base.SerializableCalculator
# https://molfeat-docs.datamol.io/stable/api/molfeat.trans.pretrained.hf_transformers.html#molfeat.trans.pretrained.hf_transformers.PretrainedHFTransformer
# https://molfeat-docs.datamol.io/stable/tutorials/add_your_own.html
# from molfeat.calc import SerializableCalculator
# class MyCalculator(SerializableCalculator):

#     def __call__(self, mol, **kwargs):
#         # you have to implement this
#         ...

#     def __len__(self):
#         # you don't have to implement this but are encouraged to do so
#         # this is used to determine the length of the output
#         ...

#     @property
#     def columns(self):
#         # you don't have to implement this
#         # use this to return the name of each entry returned by your featurizer
#         ...

#     def batch_compute(self, mols:list, **dm_parallelized_kwargs):
#         # you don't need to implement this
#         # but you should if there is an efficient batching process
#         # By default dm.parallelized arguments will also be passed as input
#         ...

