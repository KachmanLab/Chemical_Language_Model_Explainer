import os
import tqdm
import fsspec
import pickle
import warnings
import numpy as np
import pandas as pd
import datamol as dm
import matplotlib.pyplot as p
from collections import defaultdict
from rdkit.Chem import SaltRemover

from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

def load_dataset(uri: str, smiles_col: str, target_col: str):
    """Loads the MoleculeNet dataset"""
    df = pd.read_csv(uri)
    smiles = df[smiles_col].values
    y = df[target_col].values
    return smiles, y

def preprocess_smiles(smi):
    """Preprocesses the SMILES string"""
    mol = dm.to_mol(smi, ordered=True, sanitize=False)    
    try: 
        mol = dm.sanitize_mol(mol)
    except:
        mol = None
            
    if mol is None: 
        return
        
    mol = dm.standardize_mol(mol, disconnect_metals=True)
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol, dontRemoveEverything=True)

    return dm.to_smiles(mol)

def scaffold_split(smiles):
    """In line with common practice, we will use the scaffold split to evaluate our models"""
    scaffolds = [dm.to_smiles(dm.to_scaffold_murcko(dm.to_mol(smi))) for smi in smiles]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    return next(splitter.split(smiles, groups=scaffolds))

class AqueousDataMolSet():
    def __init__(self) -> None:
        smiles, y_true = load_dataset(
            "/workspace/data/AqueousSolu.csv", "smiles solute", "logS_aq_avg"
        )
        smiles = np.array([preprocess_smiles(smi) for smi in smiles])
        smiles = np.array([smi for smi in smiles if dm.to_mol(smi) is not None])
        self.smiles = smiles
        self.labels = y_true
    
    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_data = self.smiles[idx]
        if self.subset == 'train' and self.augment:
            smiles_data, _ = aug_smiles(smiles_data)
        labels = self.labels[idx]
        return smiles_data, labels
    
from rdkit import Chem 
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