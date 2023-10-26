import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from rdkit.Chem.Scaffolds import MurckoScaffold


class AqSolDataset(Dataset):
    def __init__(self, file_path, subset, split_type, split,
                 data_seed=42, augment=False):
        self.subset = subset
        self.split = split
        self.split_type = split_type
        print(split_type)
        self.data_seed = data_seed
        self.augment = augment

        # split data into accurate test set according to SolProp
        df = pd.read_csv(file_path)
        # drop one extreme outlier (logS 6.4)
        df = df[df['logS_aq_avg'] < 2.05].reset_index()
        self.min = df['logS_aq_avg'].min()
        self.max = df['logS_aq_avg'].max()

        if split_type == 'scaffold':
            splitter = MurckoScaffoldSplitter(
                smiles=df['smiles solute'].to_list(),
                n_splits=5,
                test_size=0.1,
                seed=self.data_seed
            )
        elif split_type == 'stratified':
            raise NotImplementedError
        else:
            splitter = ShuffleSplit(
                n_splits=5,
                test_size=0.1,
                random_state=self.data_seed
            )
        # split into train+val/test,
        if split_type == 'accurate':
            # predefined accurate split: >2 measuments with low std
            test_idx = np.where(
                (df['count'] > 1) & (df['logS_aq_std'] < 0.2))[0]
            _sanity = None
        else:
            _sanity, test_idx = next(
                splitter.split(df['smiles solute'].to_list()))

        if _sanity is not None:
            assert len(_sanity)+len(test_idx) == len(df)
            sane_train = df.iloc[_sanity].reset_index()
            sane_test = df.iloc[test_idx].reset_index()

        # set aside test set
        test_df = df.iloc[test_idx].reset_index()
        tr_va_df = df.drop(test_idx).reset_index()
        self.tr_va_df = tr_va_df

        # sanity check: assert train/test non-overlapping
        if _sanity is not None:
            assert tr_va_df.shape == sane_train.shape
            assert len(set(df['smiles solute'].to_list()) ^
                   set(sane_train['smiles solute'].to_list()) ^
                   set(sane_test['smiles solute'].to_list())) == 0
            assert len(set(tr_va_df['smiles solute'].to_list()) ^
                       set(sane_train['smiles solute'].to_list())) == 0
            assert len(set(test_df['smiles solute'].to_list()) ^
                       set(sane_test['smiles solute'].to_list())) == 0

        # split remaining df into train/val
        if self.subset == 'test':
            self.smiles = test_df['smiles solute'].to_list()
            self.labels = torch.tensor(
                test_df['logS_aq_avg'].to_list(), dtype=torch.float32)
        else:
            self.tr_va_splitter = splitter.split(
                tr_va_df['smiles solute'].to_list())
            self.next_split()

    def next_split(self):
        train_idx, val_idx = next(self.tr_va_splitter)

        if self.subset == 'train':
            df = self.tr_va_df.iloc[train_idx]
        elif self.subset == 'valid':
            df = self.tr_va_df.iloc[val_idx]

        self.smiles = df['smiles solute'].to_list()
        self.labels = torch.tensor(
            df['logS_aq_avg'].to_list(), dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_data = self.smiles[idx]
        if self.subset == 'train' and self.augment:
            smiles_data, _ = aug_smiles(smiles_data)
        labels = self.labels[idx]
        return smiles_data, labels


class MurckoScaffoldSplitter():
    def __init__(self, smiles, n_splits=1, test_size=0.1, seed=42, top_k=5):
        self.n_splits = n_splits
        self.test_size = test_size,
        self.seed = seed
        self.top_k = top_k
        self.topscf = self.get_top_scaffolds(
            [self.get_murcko_scaffolds(smi) for smi in smiles]
        )

    def get_murcko_scaffolds(self, smi):
        ''' get murcko-* scaffolds from smi '''
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold, canonical=True)

    def get_top_scaffolds(self, scaffolds):
        ''' get list of most frequent murcko-* scaffolds '''
        vals, cnts = np.unique(scaffolds, return_counts=True)
        # filter out scaffolds with less than 5 occurances
        return [scf for scf, cnt in list(zip(vals, cnts)) if cnt >= self.top_k]

    def split(self, smiles):
        ''' split dataset using frequent murcko-* scaffolds '''
        scaffolds = [self.get_murcko_scaffolds(smi) for smi in smiles]
        scaffolds = [s if s in self.topscf else 'rare' for s in scaffolds]
        # print('nunique', np.unique(scaffolds, return_counts=True))
        splitter = GroupShuffleSplit(n_splits=self.n_splits,
                                     test_size=0.1,
                                     random_state=self.seed)
        return splitter.split(smiles, groups=scaffolds)


class AqSolECFP(AqSolDataset):
    def __init__(self, file_path, subset, split_type, split, data_seed=42,
                 nbits=512):
        super().__init__(file_path, subset, split_type, split, data_seed)
        self.nbits = nbits
        self.make_ecfp()

    def make_ecfp(self):
        ''' TODO fix self.nbits during next_split()'''
        ecfp = [AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smi), radius=2, nBits=512  # self.nbits
               ) for smi in self.smiles]
        self.ecfp = torch.tensor(ecfp, dtype=torch.float32)

        assert len(self.ecfp) == len(self.smiles)

    def next_split(self):
        train_idx, val_idx = next(self.tr_va_splitter)

        if self.subset == 'train':
            df = self.tr_va_df.iloc[train_idx]
        elif self.subset == 'valid':
            df = self.tr_va_df.iloc[val_idx]

        self.smiles = df['smiles solute'].to_list()
        self.labels = torch.tensor(
            df['logS_aq_avg'].to_list(), dtype=torch.float32)
        self.make_ecfp()

    def __getitem__(self, idx):
        ecfp = self.ecfp[idx]
        labels = self.labels[idx]
        return ecfp, labels


class CombiSoluDataset(Dataset):
    def __init__(self, file_path, subset, temp_test, split, augment=False,
                 data_seed=42, scale_logS=True):
        self.subset = subset
        self.temp_test = temp_test
        self.augment = augment
        self.data_seed = data_seed
        self.scale_logS = scale_logS

        df = pd.read_csv(file_path)
        df = df.dropna(axis=0, subset=[
            'temperature', 'solvent_density [kg/m3]', 'experimental_logS [mol/L]'
            ])

        test_idx = round(df['temperature']) == 298
        # set aside random test set
        if not temp_test:
            ntest = round(len(df) * (1-split))
            test_idx[:ntest] = True
            test_idx[ntest:] = False
            np.random.seed(self.data_seed)
            test_idx = np.random.permutation(test_idx)

        test_df = df[test_idx]
        df = df[~test_idx]
        total = df.shape[0]
        split_index = int(total * split)

        self.min = df['experimental_logS [mol/L]'].min()
        self.max = df['experimental_logS [mol/L]'].max()
        print(self.min, self.max)

        if self.subset == 'train':
            self.solu_smi = df['solute_smiles'][:split_index].to_list()
            self.solv_smi = df['solvent_smiles'][:split_index].to_list()

            self.temperature = torch.tensor(
                df['temperature'][:split_index].to_list(), dtype=torch.float32)
            self.density = torch.tensor(
                df['solvent_density [kg/m3]'][:split_index].to_list(), dtype=torch.float32)
            self.logS = torch.tensor(
                df['experimental_logS [mol/L]'][:split_index].to_list(), dtype=torch.float32)
            if self.scale_logS:
                self.logS = self.scale(self.logS)

        elif self.subset == 'valid':
            self.solu_smi = df['solute_smiles'][split_index:].to_list()
            self.solv_smi = df['solvent_smiles'][split_index:].to_list()

            self.temperature = torch.tensor(
                df['temperature'][split_index:].to_list(), dtype=torch.float32)
            self.density = torch.tensor(
                df['solvent_density [kg/m3]'][split_index:].to_list(), dtype=torch.float32)
            self.logS = torch.tensor(
                df['experimental_logS [mol/L]'][split_index:].to_list(), dtype=torch.float32)
            if self.scale_logS:
                self.logS = self.scale(self.logS)

        elif self.subset == 'test':
            self.solu_smi = test_df['solute_smiles'].to_list()
            self.solv_smi = test_df['solvent_smiles'].to_list()

            self.temperature = torch.tensor(
                test_df['temperature'].to_list(), dtype=torch.float32)
            self.density = torch.tensor(
                test_df['solvent_density [kg/m3]'].to_list(), dtype=torch.float32)
            self.logS = torch.tensor(
                test_df['experimental_logS [mol/L]'].to_list(), dtype=torch.float32)
            if self.scale_logS:
                self.logS = self.scale(self.logS)

    def scale(self, logS):
        ''' scale logS to [0, 1] '''
        return (logS - self.min) / (self.max - self.min)

    def unscale(self, logS):
        ''' scale logS back to original range '''
        return logS * (self.max - self.min) + self.min

    def __len__(self):
        return len(self.solu_smi)

    def __getitem__(self, idx):
        solu_smi = self.solu_smi[idx]
        solv_smi = self.solv_smi[idx]
        logS = self.logS[idx]
        temperature = self.temperature[idx]
        return (solu_smi, solv_smi, temperature), logS


# from nemo_chem.data.augment import MoleculeEnumeration
def aug_smiles(smiles: str, augment_data: bool = True, canonicalize_input: bool = True):
    """Regularize SMILES by coverting to RDKit mol objects and back

    Args:
        smiles (str): Input SMILES from dataset
        canonicalize_input (bool, optional): Canonicalize by default. Defaults to False.
        smiles_augmenter: Function to augment/randomize SMILES. Defaults to None
    """
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol, canonical=True) if canonicalize_input else smiles

    if augment_data:
        # aug_mol = self.aug(mol)
        atom_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_order)
        aug_mol = Chem.RenumberAtoms(mol, atom_order) # TODO how to use PySMILESutils for this

        # There is a very rare possibility that RDKit will not be able to generate
        # the SMILES for the augmented mol. In this case we just use the canonical
        # mol to generate the SMILES
        try:
            aug_smiles = Chem.MolToSmiles(aug_mol, canonical=False)
        except RuntimeError:
            # logging.info(f'Could not generate smiles for {smiles} after augmenting. Forcing canonicalization')
            aug_smiles = canon_smiles if canonicalize_input else Chem.MolToSmiles(mol, canonical=True)
    else:
        aug_smiles = Chem.MolToSmiles(mol, canonical=False)

    assert len(aug_smiles) > 0, AssertionError('Augmented SMILES string is empty')
    assert len(canon_smiles) > 0, AssertionError('Canonical SMILES string is empty')
    return aug_smiles, canon_smiles


if __name__ == '__main__':
    import pytorch_lightning as pl
    import json
    from src.dataloader import AqSolDataset

    with open('/workspace/scripts/aqueous_config.json', 'r') as f:
        cfg = json.load(f)
    pl.seed_everything(cfg['seed'])

    cfg['split_type'] = 'scaffold'
    print('_scaffold train val test')
    train_scaffold = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train',
        cfg['split_type'], cfg['split'], data_seed=cfg['seed'], augment=False)
    valid_scaffold = AqSolDataset('/workspace/data/AqueousSolu.csv', 'valid',
        cfg['split_type'], cfg['split'], data_seed=cfg['seed'], augment=False)
    test_scaffold = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test',
        cfg['split_type'], cfg['split'], data_seed=cfg['seed'], augment=False)

    cfg['split_type'] = 'accurate'
    print('train_accurate')
    train_accurate = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train',
        cfg['split_type'], cfg['split'], data_seed=cfg['seed'], augment=False)

    cfg['split_type'] = 'random'
    print('train_random')
    train_random = AqSolDataset('/workspace/data/AqueousSolu.csv', 'valid',
        cfg['split_type'], cfg['split'], data_seed=cfg['seed'], augment=False)

    cfg['split_type'] = 'scaffold'
    sanity = AqSolDataset('/workspace/data/AqueousSolu.csv', 'train',
        cfg['split_type'], cfg['split'], data_seed=cfg['seed'], augment=False)

    assert len(sanity.smiles) == len(train_scaffold.smiles)
    assert set(sanity.smiles) == set(train_scaffold.smiles)
    assert set(sanity.smiles) ^ set(train_scaffold.smiles) == set()

    ecfp_scaffold = AqSolECFP('/workspace/data/AqueousSolu.csv', 'train',
        'scaffold', cfg['split'], data_seed=cfg['seed'], nbits=512)
    ecfp_random = AqSolECFP('/workspace/data/AqueousSolu.csv', 'valid',
        'random', cfg['split'], data_seed=cfg['seed'], nbits=512)
    ecfp_accurate = AqSolECFP('/workspace/data/AqueousSolu.csv', 'test',
        'accurate', cfg['split'], data_seed=cfg['seed'], nbits=512)

    train_scaffold.next_split()
    valid_scaffold.next_split()
    sanity.next_split()
    ecfp_scaffold.next_split()
    ecfp_random.next_split()

    assert train_scaffold.smiles[42] == sanity.smiles[42]
    assert ecfp_scaffold.smiles[42] == train_scaffold.smiles[42]
