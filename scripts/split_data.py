import pytorch_lightning as pl
from src.dataloader import AqSolDataset
import pickle
import pandas as pd
import dvc.api

cfg = dvc.api.params_show()
print('ds', cfg['ml']['model'], cfg['ml']['head'])
print('ml', cfg['ds']['task'], cfg['ds']['split'])

pl.seed_everything(cfg['ml']['seed'])
root = f"/workspace/data/{cfg['ds']['task']}/{cfg['ds']['split']}"

train_ds = AqSolDataset(
    cfg['ds']['filepath'], 'train', cfg['ds']['split'],
    cfg['ds']['split_frac'], cfg['ds']['n_splits'], cfg['ds']['data_seed'],
    augment=False)
valid_ds = AqSolDataset(
    cfg['ds']['filepath'], 'valid', cfg['ds']['split'],
    cfg['ds']['split_frac'], cfg['ds']['n_splits'], cfg['ds']['data_seed'])
test_ds = AqSolDataset(
    cfg['ds']['filepath'], 'test', cfg['ds']['split'],
    cfg['ds']['split_frac'], cfg['ds']['n_splits'], cfg['ds']['data_seed'])

test = test_ds[:]
with open(f"{root}/test.pkl", 'wb') as f:
    pickle.dump(test, f)

for fold in range(cfg['ds']['n_splits']):
    train = train_ds[fold]
    with open(f"{root}/train{fold}.pkl", 'wb') as f:
        pickle.dump(train, f)

for fold in range(cfg['ds']['n_splits']):
    valid = valid_ds[fold]
    with open(f"{root}/valid{fold}.pkl", 'wb') as f:
        pickle.dump(valid, f)


# write split into csv for reproducibility
propname = cfg['ds']['propname']
results = pd.DataFrame(columns=['SMILES', propname, 'Split'])
for subset, data in list(zip(['test', 'val', 'train'], [test, valid, train])):
    # reverse order for consistency with plotting
    # smiles = list(chain(*[fet('smiles') for f in all]))
    # labels = torch.concat([f.get('labels') for f in all]).cpu().numpy()

    res = pd.DataFrame({
        'SMILES': list(data.smiles),
        propname: data.labels.numpy(),
        'Subset': subset
        })
    results = pd.concat([results, res], axis=0)

# reset index to correspond to visualization UID
results = results.reset_index(drop=True)
results = results.reset_index().rename(columns={'index': 'uid'})
results.to_csv(f"{root}/{cfg['ds']['split']}_df.csv")
