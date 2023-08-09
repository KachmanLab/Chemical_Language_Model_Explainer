import pandas as pd
import numpy as np

def unf(st):
    st = st.strip("'[").strip("']").split(' ')
    st = [s for s in st if s not in ['']]
    return np.array(st, dtype=float)

df_logp = pd.read_csv('results/logp/logp_predictions.csv')
df_logp['crippen_weights'] = df_logp['crippen_weights'].apply(unf)
df_logp['crippen_raw'] = df_logp['crippen_raw'].apply(unf)
df_logp['ours_weights'] = df_logp['ours_weights'].apply(unf)
df_logp['crippen_rmse'] = np.sqrt((df_logp['logp_crippen'] - df_logp['logp_exp']) ** 2)

df_shap = pd.read_csv('results/logp/logp_shap_predictions.csv')
df_shap['shap_weights'] = df_shap['shap_weights'].apply(unf)

assert max([max(x) for x in df_logp['ours_weights']]) <= 1
assert max([max(x) for x in df_logp['crippen_weights']]) <= 1
assert max([max(x) for x in df_shap['shap_weights']]) <= 1

# Merge the dataframes based on the 'UID' column
merged = pd.merge(df_logp[['uid', 'smiles', 'crippen_raw', 'crippen_weights',
                           'ours_weights', 'crippen_rmse']], 
                  df_shap[['uid', 'smiles', 'shap_raw', 'shap_weights']],                   
                  on='uid', 
                  suffixes=('_logp', '_shap'))

if not all(merged['smiles_logp'] == merged['smiles_shap']):
    raise ValueError("SMILES values are different between the CSV files.")

# Drop the duplicate SMILES column after validation
merged.drop('smiles_shap', axis=1, inplace=True)
merged.rename(columns={'SMILES_logp': 'SMILES'}, inplace=True)
merged.to_csv('results/logp/all_predictions.csv', index=False)

def mrmse(pred_1, pred_2):
    """
    evaluate mean RMSE between two sets of attributed relevance weights
    """
    if len(pred_1) != len(pred_2):
        raise ValueError("relevances must have the same number of vectors.")
    
    rmse_values = []
    for p, t in zip(pred_1, pred_2):
        if len(p) != len(t):
            raise ValueError("Each pair of relevances vectors must have the same length.")
        
        squared_errors = (p - t) ** 2
        rmse = np.sqrt(np.mean(squared_errors))
        rmse_values.append(rmse)
    
    return np.mean(rmse_values), np.std(rmse_values)

ours_shap, ours_shap_sd = mrmse(
    merged['ours_weights'], merged['shap_weights']
)
ours_crip, ours_crip_sd = mrmse(
    merged['ours_weights'], merged['crippen_weights']
)
shap_crip, shap_crip_sd = mrmse(
    merged['shap_weights'], merged['crippen_weights']
)

print(f"Ours vs SHAP: mean {ours_shap:.4f}, \tsd {ours_shap_sd:.4f}")
print(f"Ours vs Crip: mean {ours_crip:.4f}, \tsd {ours_crip_sd:.4f}")
print(f"SHAP vs Crip: mean {shap_crip:.4f}, \tsd {shap_crip_sd:.4f}")

idx = np.where(merged['crippen_rmse'] <= 1.3, True, False)

ours_shap, ours_shap_sd = mrmse(
    merged['ours_weights'][idx], merged['shap_weights'][idx]
)
ours_crip, ours_crip_sd = mrmse(
    merged['ours_weights'][idx], merged['crippen_weights'][idx]
)
shap_crip, shap_crip_sd = mrmse(
    merged['shap_weights'][idx], merged['crippen_weights'][idx]
)

print(f"Ours vs SHAP: mean {ours_shap:.4f}, \tsd {ours_shap_sd:.4f}")
print(f"Ours vs Crip: mean {ours_crip:.4f}, \tsd {ours_crip_sd:.4f}")
print(f"SHAP vs Crip: mean {shap_crip:.4f}, \tsd {shap_crip_sd:.4f}")

