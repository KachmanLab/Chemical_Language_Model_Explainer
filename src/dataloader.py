import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import datamol as dm

class AqSolDataset(Dataset):
    def __init__(self, file_path, subset, split_type, split,
                 data_seed=42, augment=False):
        self.subset = subset
        self.split = split
        self.split_type = split_type
        self.data_seed = data_seed
        self.augment = augment

        # split data into accurate test set according to SolProp specifications
        df = pd.read_csv(file_path)
        if split_type == 'accurate':
            test_idx = (df['count'] > 1) & (df['logS_aq_std'] < 0.2)
        elif split_type == 'random':
            # TODO fix train/val 10% even split for random
            ntest = round(len(df) * (1-split))
            test_idx[:ntest] = True
            test_idx[ntest:] = False
            np.random.seed(self.data_seed)
            test_idx = np.random.permutation(test_idx)
        elif split_type == 'scaffold':
            # TODO split into 80/10/10, get ID over smiles,
            # fit into train/val/test setting below
            scaffold_split(df['smiles solute'].to_list())
            raise NotImplementedError

                #df.smiles

        # WIP get all incides
        # WIP assert indices non-overlapping, total=len(df)
        # if accurate:
        #     - remove test
        #     - randomShuffleSplit(train, val)
        # elif random:
        #     - randomShuffleSplit(train, val, test)
        # elif scaffold:
        #     - randomShuffleSplit(train, val, test)

        test_df = df[test_idx]
        df = df[~test_idx]
        total = df.shape[0]
        split_index = int(total * split)

        # drop one extreme outlier (logS 6.4)
        df = df[df['logS_aq_avg'] < 2.05]
        self.min = df['logS_aq_avg'].min()
        self.max = df['logS_aq_avg'].max()

        if self.subset == 'train':
            self.smiles = df['smiles solute'][:split_index].to_list()
            labels = df['logS_aq_avg'][:split_index].to_list()
            self.labels = torch.tensor(labels, dtype=torch.float32)

        elif self.subset == 'valid':
            self.smiles = df['smiles solute'][split_index:].to_list()
            val_labels = df['logS_aq_avg'][split_index:].to_list()
            self.labels = torch.tensor(val_labels, dtype=torch.float32)

        elif self.subset == 'test':
            self.smiles = test_df['smiles solute'].to_list()
            test_labels = test_df['logS_aq_avg'].to_list()
            self.labels = torch.tensor(test_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_data = self.smiles[idx]
        if self.subset == 'train' and self.augment:
            smiles_data, _ = aug_smiles(smiles_data)
        labels = self.labels[idx]
        return smiles_data, labels


class AqSolECFP(AqSolDataset):
    def __init__(self, file_path, subset, split_type, split, data_seed=42,
                 nbits=512):
        super().__init__(file_path, subset, split_type, split, data_seed)

        ecfp = [AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smi), radius=2, nBits=nbits
               ) for smi in self.smiles]
        self.ecfp = torch.tensor(ecfp, dtype=torch.float32)

        assert len(self.ecfp) == len(self.smiles)

    def __getitem__(self, idx):
        ecfp = self.ecfp[idx]
        labels = self.labels[idx]
        return ecfp, labels

    #     loader = dc.data.CSVLoader(
    #                 tasks = ['logS_aq_avg'],
    #                 # id_field = "smiles solute",
    #                 feature_field = "smiles solute",
    #                 featurizer = dc.feat.CircularFingerprint(),
    #                 # featurizer = 'ECFP'
    #     )
    #     dataset = loader.create_dataset(file_path)
    #     print(type(dataset))
    #     print(dataset)
    #     return dataset#.load_dataset('/workspace/data/AqueousSolu-Exp.csv')
    # 

def scaffold_split(smiles, subset):
    """In line with common practice, we will use the scaffold split to evaluate our models"""
    scaffolds = [dm.to_smiles(dm.to_scaffold_murcko(dm.to_mol(smi))) for smi in smiles]
    splitter = GroupShuffleSplit(n_splits=2, valid_size=0.1,
                                 test_size=0.1, random_state=cfg['seed'])
    train, val, test = splitter.split(smiles, groups=scaffolds)
    return train, val, test
    #return next(splitter.split(smiles, groups=scaffolds))



class CombiSoluDataset(Dataset):
    def __init__(self, file_path, subset, temp_test, split, augment=False, 
            data_seed=42, scale_logS=True):
        self.subset = subset
        self.temp_test = temp_test
        self.augment = augment
        self.data_seed = data_seed
        self.scale_logS = scale_logS

        df = pd.read_csv(file_path)        
        df = df.dropna(axis=0, 
            subset=['temperature', 'solvent_density [kg/m3]', 'experimental_logS [mol/L]'])
        
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
