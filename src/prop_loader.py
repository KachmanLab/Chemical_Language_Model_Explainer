import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import pandas as pd
import numpy as np

class LogPDataset(Dataset):
    def __init__(self, file_path, subset, split, data_seed=42, augment=False):
        self.subset = subset
        self.data_seed = data_seed
        self.augment = augment # not implemented

        # load & split data into accurate test set according to SolProp specifications
        df = pd.read_csv(file_path)
        df = df[~pd.isna(df['LogP'])]

        #test_idx = (df['LogP'] > -4.) & (df['LogP'] < 6.3)
        test_idx = (df['LogP'] > -20.)
        ntest = round(len(df) * (1-split))
        test_idx[:ntest] = True
        test_idx[ntest:] = False
        np.random.seed(self.data_seed)
        test_idx = np.random.permutation(test_idx)

        test_df = df[test_idx]
        df = df[~test_idx]
        total = df.shape[0]
        split_index = int(total * split)

        # df = df[df['LogP'] < 2.05]
        self.min = df['LogP'].min()
        self.max = df['LogP'].max()
        print('minmax', self.min, self.max)

        if self.subset == 'train':
            self.smiles = df['smiles'][:split_index].to_list()
            labels = df['LogP'][:split_index].to_list()
            self.labels = torch.tensor(labels, dtype=torch.float32)

        elif self.subset == 'valid':
            self.smiles = df['smiles'][:split_index].to_list()
            val_labels = df['LogP'][:split_index].to_list()
            self.labels = torch.tensor(val_labels, dtype=torch.float32)

        elif self.subset == 'test':
            self.smiles = test_df['smiles'][:split_index].to_list()
            test_labels = test_df['LogP'][:split_index].to_list()
            self.labels = torch.tensor(test_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_data = self.smiles[idx]
        labels = self.labels[idx]
        return smiles_data, labels
