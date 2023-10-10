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
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    return next(splitter.split(smiles, groups=scaffolds))

class AqueousDataMolSet():
    def __init__(self) -> tuple:
        smiles, y_true = load_dataset(
            "/workspace/data/AqueousSolu.csv", "smiles solute", "logS_aq_avg"
        )
        smiles = np.array([preprocess_smiles(smi) for smi in smiles])
        smiles = np.array([smi for smi in smiles if dm.to_mol(smi) is not None])
        return smiles, y_true