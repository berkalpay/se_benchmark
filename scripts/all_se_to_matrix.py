"""
Converts list of drug-SE pairs to a matrix.
"""

import pandas as pd

with open("data/SIDER/meddra_all_se.tsv") as f:
    lines = [l.split("\t") for l in f.read().splitlines()]
    
chem_ids = list(set(l[0] for l in lines))
se_ids = list(set(l[2] for l in lines))
df = pd.DataFrame(0, index=chem_ids, columns=se_ids)
df_dict = df.to_dict()
for l in lines:
    df_dict[l[2]][l[0]] = 1

df = pd.DataFrame(df_dict)
df = df.loc[sorted(chem_ids), sorted(se_ids)]
df.to_csv("data/SIDER/cid_umls_matrix.tsv", sep="\t")
