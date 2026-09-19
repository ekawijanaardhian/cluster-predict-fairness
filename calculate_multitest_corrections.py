import pandas as pd
import numpy as np

df_boot = pd.read_csv('results/statistical_bootstrap_inference.csv')
deltas = df_boot[df_boot['Architecture'].str.startswith('Delta_')].copy()

records = []
for idx, row in deltas.iterrows():
    raw_str = row['Formatted_95CI']
    p_str = raw_str.split('(p=')[1].replace(')', '')
    p_val = float(p_str)
    records.append({
        'Architecture': row['Architecture'],
        'Metric': row['Metric'],
        'Point_Estimate': row['Point_Estimate'],
        'CI_Lower': row['CI_Lower'],
        'CI_Upper': row['CI_Upper'],
        'Raw_P_Value': p_val
    })

df_res = pd.DataFrame(records)

m = len(df_res)

df_res = df_res.sort_values(by='Raw_P_Value').reset_index(drop=True)

holm_p = []
curr_max = 0.0
for i, p in enumerate(df_res['Raw_P_Value']):
    adj = (m - i) * p
    curr_max = max(curr_max, adj)
    holm_p.append(min(1.0, curr_max))
df_res['Holm_P_Value'] = holm_p

bh_p = [0.0] * m
curr_min = 1.0
for i in range(m - 1, -1, -1):
    k = i + 1
    p = df_res.loc[i, 'Raw_P_Value']
    adj = (m / k) * p
    curr_min = min(curr_min, adj)
    bh_p[i] = min(1.0, curr_min)
df_res['BH_FDR_P_Value'] = bh_p

df_res['Sig_Holm_05'] = df_res['Holm_P_Value'] < 0.05
df_res['Sig_FDR_05'] = df_res['BH_FDR_P_Value'] < 0.05

print("=" * 100)
print("MULTIPLE TESTING CORRECTIONS (BOOTSTRAP P-VALUES, M=12 COMPARISONS):")
print("=" * 100)
print(df_res.to_string(index=False))

df_delong = pd.read_csv('results/delong_auc_hypothesis_tests.csv')
print("\n" + "=" * 100)
print("DELONG AUC TESTS (M=5 COMPARISONS):")
print("=" * 100)
m_delong = len(df_delong)
df_delong = df_delong.sort_values(by='DeLong_P_Value').reset_index(drop=True)
holm_delong = []
curr_max = 0.0
for i, p in enumerate(df_delong['DeLong_P_Value']):
    adj = (m_delong - i) * p
    curr_max = max(curr_max, adj)
    holm_delong.append(min(1.0, curr_max))
df_delong['Holm_P_Value'] = holm_delong

bh_delong = [0.0] * m_delong
curr_min = 1.0
for i in range(m_delong - 1, -1, -1):
    k = i + 1
    p = df_delong.loc[i, 'DeLong_P_Value']
    adj = (m_delong / k) * p
    curr_min = min(curr_min, adj)
    bh_delong[i] = min(1.0, curr_min)
df_delong['BH_FDR_P_Value'] = bh_delong
print(df_delong.to_string(index=False))
