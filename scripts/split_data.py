import pytorch_lightning as pl
import json
from src.dataloader import AqSolDataset
import pickle
import pandas as pd

with open('/workspace/cfg/data_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['data_seed'])
root = f"/workspace/data/{cfg['property']}_proc/{cfg['split_type']}"

train_ds = AqSolDataset(cfg['filepath'], 'train', cfg['split'],
                        cfg['split_frac'], cfg['n_splits'], cfg['data_seed'],
                        augment=False)
valid_ds = AqSolDataset(cfg['filepath'], 'valid', cfg['split'],
                        cfg['split_frac'], cfg['n_splits'], cfg['data_seed'])
test_ds = AqSolDataset(cfg['filepath'], 'test', cfg['split'],
                       cfg['split_frac'], cfg['n_splits'], cfg['data_seed'])

test = test_ds[:]
with open(f"{root}/test.pkl", 'wb') as f:
    pickle.dump(test, f)

for fold in range(cfg['n_splits']):
    train = train_ds[fold]
    with open(f"{root}/train{fold}.pkl", 'wb') as f:
        pickle.dump(train, f)

for fold in range(cfg['n_splits']):
    valid = valid_ds[fold]
    with open(f"{root}/valid{fold}.pkl", 'wb') as f:
        pickle.dump(valid, f)


# write split into csv for reproducibility
results = pd.DataFrame(columns=['SMILES', 'logS_exp', 'Split'])
for subset, data in list(zip(['test', 'val', 'train'], [test, valid, train])):
    # reverse order for consistency with plotting
    # smiles = list(chain(*[fet('smiles') for f in all]))
    # labels = torch.concat([f.get('labels') for f in all]).cpu().numpy()

    res = pd.DataFrame({
        'SMILES': list(data.smiles),
        'logS_exp': data.labels.numpy(),
        'Subset': subset
        })
    results = pd.concat([results, res], axis=0)

# reset index to correspond to visualization UID
results = results.reset_index(drop=True)
results = results.reset_index().rename(columns={'index': 'uid'})
results.to_csv(f"{root}/{cfg['split']}_df.csv")
