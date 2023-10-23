import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np


class CMCDataset(Dataset):
    def __init__(self, file_path, subset, split, data_seed=42, augment=False):
        self.subset = subset
        self.data_seed = data_seed
        self.augment = augment  # not implemented

        # load & split data
        df = pd.read_csv(file_path)
        df = df[~pd.isna(df['p(CMC)'])]
        df = df[(df['p(CMC)'] > -6.6) & (df['p(CMC)'] < 5.5)]

        # drop line 1173: logCMC = 23.3
        # CCCCCCCCCCOCC(COCCC[N+](C)(C)C)(COCCCCCCCCCC)COCCC[N+](C)(C)C.[I-].[I-]

        #test_idx = (df['p(CMC)'] > -4.) & (df['p(CMC)'] < 6.3)
        test_idx = (df['p(CMC)'] < np.inf)
        ntest = round(len(df) * (1-split))
        test_idx[:ntest] = True
        test_idx[ntest:] = False
        np.random.seed(self.data_seed)
        test_idx = np.random.permutation(test_idx)

        test_df = df[test_idx]
        df = df[~test_idx]
        total = df.shape[0]
        split_index = int(total * split)

        # df = df[df['p(CMC)'] < 2.05]
        self.min = df['p(CMC)'].min()
        self.max = df['p(CMC)'].max()
        print('min', self.min, 'max', self.max)

        if self.subset == 'train':
            self.smiles = df['SMILES'][:split_index].to_list()
            labels = df['p(CMC)'][:split_index].to_list()
            self.labels = torch.tensor(labels, dtype=torch.float32)

        elif self.subset == 'valid':
            self.smiles = df['SMILES'][:split_index].to_list()
            val_labels = df['p(CMC)'][:split_index].to_list()
            self.labels = torch.tensor(val_labels, dtype=torch.float32)

        elif self.subset == 'test':
            self.smiles = test_df['SMILES'][:split_index].to_list()
            test_labels = test_df['p(CMC)'][:split_index].to_list()
            self.labels = torch.tensor(test_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_data = self.smiles[idx]
        labels = self.labels[idx]
        return smiles_data, labels


class CMCECFP(CMCDataset):
    def __init__(self, file_path, subset, split, data_seed=42,
                 nbits=512):
        super().__init__(file_path, subset, split, data_seed)

        ecfp = [AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smi), radius=2, nBits=nbits
               ) for smi in self.smiles]
        self.ecfp = torch.tensor(ecfp, dtype=torch.float32)

        assert len(self.ecfp) == len(self.smiles)

    def __getitem__(self, idx):
        ecfp = self.ecfp[idx]
        labels = self.labels[idx]
        return ecfp, labels
