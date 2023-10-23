from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json
from pytorch_lightning.loggers import WandbLogger
from src.dataloader import AqSolDataset
from src.model import AqueousRegModel, BaselineAqueousModel

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])

train_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train',
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'], augment=False)
val_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'valid',
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test',
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])

train_loader = DataLoader(train_dataset, batch_size=cfg['n_batch'],
                          shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=cfg['n_batch'],
                        shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=cfg['n_batch'],
                         shuffle=False, num_workers=8)

if cfg['model'] == 'mmb':
    print(cfg['head'])
    model = AqueousRegModel(head=cfg['head'])
elif cfg['model'] == 'shap':
    model = BaselineAqueousModel()

if cfg['finetune']:
    # unfreeze to train the whole model instead of just the head
    model.mmb.unfreeze()

wandb_logger = WandbLogger(project='aqueous-solu',
                           #dir='/results/aqueous/models/'
                           )
wandb_logger.experiment.config.update(cfg)

trainer = pl.Trainer(
    max_epochs=cfg['n_epochs'],
    accelerator='gpu',
    gpus=1,
    precision=16,
    logger=wandb_logger,
    auto_lr_find=True,
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

#mdir = 'aqueous' if cfg['model'] == 'mmb' else 'shap'
mdir = 'mmb' if cfg['model'] == 'mmb' else 'shap'
modelpath = f'/workspace/results/aqueous/models/aqueous_{mdir}.pt'
trainer.save_checkpoint(modelpath)
