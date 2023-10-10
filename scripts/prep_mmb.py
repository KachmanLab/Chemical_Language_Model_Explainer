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
# smiles = smiles[:32]

print('loading mmb')
model = MMBFeaturizer()
feats_mmb = model.featurize(smiles)
print(feats_mmb.shape)

all_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train', False, 
    0., data_seed=cfg['seed'], augment=False)
all_loader = DataLoader(all_dataset, batch_size=cfg['n_batch'], 
    shuffle=False, num_workers=8)

trainer = pl.Trainer(
    accelerator='cpu',
    gpus=1,
    precision=16,
)
feats_mmb = trainer.predict(model, all_loader)
feats_mmb = torch.cat(feats_mmb).numpy()
print(feats_mmb.shape)


# Setup the featurizers
trans_ecfp = FPVecTransformer(kind="ecfp:4", length = 512,
                              n_jobs=-1, dtype=np.float32)
feats_ecfp, ind_ecfp = trans_ecfp(smiles, ignore_errors=True)
print(feats_ecfp.shape)
print(ind_ecfp)
print(ind_ecfp.shape)

reps = {
    "smiles": smiles,
    "target": y_true,
    "mmb": feats_mmb, #[ind_mmb]
    "ecfp": feats_ecfp[ind_ecfp],
}

print(reps)

pickle.dump(reps, '/workspace/data/prep/aqueous_mmb.pickle',)

train_ind, test_ind = scaffold_split(smiles)
print(smiles[test_ind][:5])







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

