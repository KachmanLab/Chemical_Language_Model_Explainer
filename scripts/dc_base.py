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

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

dataset = AqSolDeepChem('/workspace/data/AqueousSolu.csv', 'all', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])

# split dataset
splitter = dc.splits.SingletaskStratifiedSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset=dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1
)
# train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'], 
#     shuffle=True, num_workers=8)


# MultitaskFitTransformRegressor
regressor = dc.models.MultitaskRegressor(
    n_tasks = 1,
    n_features = 512,
    layer_sizes = [1],
    dropouts = 0.,
)


metric = dc.metrics.Metric(dc.metrics.mae_score)
# metrics = [dc.metrics.Metric(dc.metrics.mae_score),
#             dc.metrics.Metric(dc.metrics.rmse_score)]
# metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")


from sklearn.ensemble import RandomForestRegressor
skmodel = RandomForestRegressor(n_estimators=500)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
skmodel = LinearRegression()

model = dc.models.SklearnModel(model=skmodel)

# model training
model.fit(train_dataset)
valid_preds = model.predict(valid_dataset)
valid_preds.shape

# evaluate the model
train_score = model.evaluate(train_dataset, [metric])
valid_score = model.evaluate(valid_dataset, [metric])
test_score = model.evaluate(test_dataset, [metric])

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