import pytorch_lightning as pl
# from src.dataloader import *
import pickle
import pandas as pd
import dvc.api
import hydra
from omegaconf import DictConfig, OmegaConf
import importlib

@hydra.main(
    version_base="1.3", config_path="/workspace/conf", config_name="config")
def split(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.load('/workspace/params.yaml')
    print('SPLIT CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.split.data_seed)
    root = f"/workspace/data/{cfg.task.task}/{cfg.split.split}"

    module = importlib.import_module('src.dataloader')
    DatasetLoader = getattr(module, cfg.task.loader)

    train_ds = DatasetLoader('train',
        cfg.task.filepath, cfg.task.smilesname, cfg.task.propname,
        cfg.split.split, cfg.split.split_frac, cfg.split.n_splits,
        cfg.split.data_seed, augment=False)
    valid_ds = DatasetLoader('valid',
        cfg.task.filepath, cfg.task.smilesname, cfg.task.propname,
        cfg.split.split, cfg.split.split_frac, cfg.split.n_splits,
        cfg.split.data_seed)
    test_ds = DatasetLoader('test', 
        cfg.task.filepath, cfg.task.smilesname, cfg.task.propname,
        cfg.split.split, cfg.split.split_frac, cfg.split.n_splits,
        cfg.split.data_seed)

    test = test_ds[:]
    print(len(test), 'test len')
    with open(f"{root}/test.pkl", 'wb') as f:
        pickle.dump(test, f)

    for fold in range(cfg.split.n_splits):
        train = train_ds[fold]
        with open(f"{root}/train{fold}.pkl", 'wb') as f:
            pickle.dump(train, f)
            print(len(train), 'train len', fold)

    for fold in range(cfg.split.n_splits):
        valid = valid_ds[fold]
        with open(f"{root}/valid{fold}.pkl", 'wb') as f:
            pickle.dump(valid, f)
            print(len(valid), 'valid len', fold)

    # write split into csv for reproducibility
    propname = cfg.task.propname
    results = pd.DataFrame(columns=['SMILES', propname, 'Split'])
    for subset, data in list(zip(
            ['test', 'valid', 'train'], [test, valid, train])):
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
    results.to_csv(f"{root}/{cfg.split.split}_df.csv")


if __name__ == "__main__":
    import dvc.api
    dvccfg = DictConfig(dvc.api.params_show())
    print(dvccfg)
    split()
