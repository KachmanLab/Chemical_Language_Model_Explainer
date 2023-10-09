import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
import dagshub
# from dagshub.pytorch_lightning import DAGsHubLogger
import mlflow
from src.dataloader import AqSolDataset, AqSolDeepChem
from src.model import AqueousRegModel, BaselineAqueousModel
import deepchem as dc
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# from deepchem.models.torch_models.grover import GroverModel
# from deepchem.feat.vocabulary_builders import (GroverAtomVocabularyBuilder, GroverBondVocabularyBuilder)

# with open('/workspace/scripts/aqueous_config.json', 'r') as f:
with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

# dataset = AqSolDeepChem('/workspace/data/AqueousSolu.csv', 'all', 
    # cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])
# train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'], 
#    shuffle=True, num_workers=8)

# https://deepchem.readthedocs.io/en/latest/api_reference/models.html
featurizer = {
        'ECFP': dc.feat.CircularFingerprint(),
        'DMPNN': dc.feat.DMPNNFeaturizer(
            features_generators=["rdkit_desc_normalized"]
        ),
        'AttentiveFP': dc.feat.MolGraphConvFeaturizer(),
        'GraphConv': dc.feat.ConvMolFeaturizer(),
        # 'Mordred': dc.feat.MordredDescriptors(),
        # 'Chemberta': dc.feat.RobertaTokenizer()
        # 'Grover': dc.feat.GroverFeaturizer(),
    }.get(cfg['featurizer'])

if cfg["dataset"] == "logs":
    ## log(S) ###
    loader = dc.data.CSVLoader(
            tasks = ['logS_aq_avg'],
            feature_field = "smiles solute",
            featurizer = featurizer)
    dataset = loader.create_dataset('/workspace/data/AqueousSolu.csv',)

elif cfg["dataset"] == "cmc":
    ### CMC ###
    loader = dc.data.CSVLoader(
            tasks = ['p(CMC)'],
            feature_field = "SMILES",
            featurizer = featurizer)
    dataset = loader.create_dataset('/workspace/data/cmc_dataset_1235.csv',)

# split dataset
splitter = dc.splits.SingletaskStratifiedSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset=dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
)

print(f"train_dataset {len(train_dataset)}")
print(f"{train_dataset.X.shape}")
print(test_dataset.ids[:2])


metrics = [
    dc.metrics.Metric(dc.metrics.mae_score), 
    dc.metrics.Metric(dc.metrics.mean_squared_error),
    dc.metrics.Metric(dc.metrics.rms_score),
    ]

if cfg['model'] == 'Reg':
    regressor = dc.models.MultitaskFitTransformRegressor(
        n_tasks = 1,
        # n_features = 512,
        # layer_sizes = [1],
        n_features = train_dataset.X.shape[1] or 2048,
        layer_sizes = [1024, 512, 256, 1],
        dropouts = 0.2,
    )
    model = regressor

elif cfg['model'] in ['RF', 'SVR', 'Linear', 'Lasso', 'Ridge']:
    model = {
        'RF': RandomForestRegressor(n_estimators=500),
        'SVR': SVR(),
        'Linear': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
    }.get(cfg['model'], LinearRegression())
    model = dc.models.SklearnModel(model=model)

elif cfg['model'] == 'DMPNN':
    model = dc.models.DMPNNModel(
        mode="regression", n_tasks=1
    )   
elif cfg['model'] == 'AttentiveFP': 
    model = dc.models.AttentiveFPModel(
       mode="regression", n_tasks=1 
    )
else: 
    raise NotImplementedError, f'{cfg["model"]} not implemented'
# elif cfg['model'] == 'GraphConv':
#     from deepchem.models.graph_models import GraphConvModel, WeaveModel
#     model = GraphConvModel(
#         mode="regression", n_tasks=1
#     )

# model training
print(f"Starting fit of {cfg['model']} using {cfg['featurizer']}")
model.fit(train_dataset)#, nb_epoch=100)
valid_preds = model.predict(valid_dataset)
valid_preds.shape

# evaluate the model
train_score = model.evaluate(train_dataset, metrics)
valid_score = model.evaluate(valid_dataset, metrics)
test_score = model.evaluate(test_dataset, metrics)

print(f"train_score {train_score}")
print(f"valid_score {valid_score}")
print(f"test_score {test_score}")
print(f"Results of model {cfg['model']} with {cfg['featurizer']}")
# ft_model = AqueousRegModel()



# dagshub.init("Chemical_Language_Model_Explainer", "stefanhoedl", mlflow=True)

# trainer = pl.Trainer(
#     max_epochs=cfg['n_epochs'],
#     accelerator='gpu',
#     gpus=1,
#     precision=16,
#     auto_lr_find=True,
# )

# with mlflow.start_run() as run:
#     mlflow.pytorch.autolog(log_models=False)
#     mlflow.log_params(cfg)

#     trainer.fit(ft_model, train_loader, val_loader)
#     trainer.test(ft_model, test_loader)

#     mdir = 'aqueous' if cfg['model'] == 'reg' else 'shap'
#     modelpath = f'/workspace/results/{mdir}/models/aqueous_{run.info.run_id}.pt'
#     trainer.save_checkpoint(modelpath)
#     # mlflow.log_artifact(modelpath)