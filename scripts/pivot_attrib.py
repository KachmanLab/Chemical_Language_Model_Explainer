import torch
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import hydra
import json
from omegaconf import OmegaConf, DictConfig
import numpy as np
import seaborn as sns
import os
import json
from sklearn.metrics.pairwise import cosine_similarity


def restore_array(st):
    st = st.strip("'[").strip("']").split(' ')
    st = [s for s in st if s not in ['']]
    return np.array(st, dtype=float)


def clean_string(st):
    tokens = st.strip("[").strip("]").split(',')
    return [t.replace("'", "").replace(" ", "") for t in tokens]


def explode_attribs(models=None):
    cfg = OmegaConf.load('./params.yaml')
    print('PIVOT CONFIG from params.yaml')
    print(OmegaConf.to_yaml(cfg))

    basepath = f"/workspace/final/{cfg.task.task}/{cfg.split.split}"
    if not models:
        models = [
            'mmb-ft-lin', 'mmb-ft-hier',
            'mmb-lin', 'mmb-hier',
            'mmb-ft-avg-lin', 'mmb-ft-avg-hier',
            'mmb-avg-lin', 'mmb-avg-hier',
            'ecfp-lin', 'ecfp2k-lin',
            'ecfp-lin-scaled', 'ecfp2k-lin-scaled',
            'ecfp-hier', 'ecfp2k-hier',
            'ecfp-svr', 'ecfp2k-svr',
            'ecfp-rf', 'ecfp2k-rf',
        ]

    for mdir in models:
        try:
            print(mdir)
            df = pd.read_csv(f"{basepath}/{mdir}/attributions.csv")
            if 'avg' in mdir:
                # SHAP for -avg-
                df['rel_weights'] = df['shap_weights'].apply(restore_array)
                df['uid'] = range(len(df))
                df['tokens'] = df['tokens'].apply(clean_string)
            elif 'ecfp' in mdir:
                # ECFP
                df['rel_weights'] = df['atom_weights'].apply(restore_array)
            else:
                # <R> for ours
                df['rel_weights'] = df['rel_weights'].apply(restore_array)
                df['tokens'] = df['tokens'].apply(clean_string)

            # explode (pivot) from one uid per row to one token-weight per row
            if 'ecfp' in mdir:
                df = df.loc[:, ['uid', 'rel_weights', 'smiles']]
                dfexp = df.explode(['rel_weights'])
            else:
                # extract relevant columns
                df = df.loc[:, ['uid', 'tokens', 'rel_weights']]
                dfexp = df.explode(['rel_weights', 'tokens'])

            # cleanup
            dfexp = dfexp.reset_index(drop=True)
            dfexp = dfexp.rename(columns={'tokens': 'token', 'rel_weights': 'weight'})
            dfexp.to_csv(f"{basepath}/attribs/{mdir}_attribs.csv", index=False)

        except FileNotFoundError:
            print(f"File not found: {basepath}/{mdir}")
            continue


if __name__ == "__main__":
    explode_attribs()
