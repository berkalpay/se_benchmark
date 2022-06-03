"""
Converts side effects in the SIDER matrix to the MedDRA ontology.
Creates a data set for each level of the MedDRA hierarchy.
"""

import sys
import pandas as pd
import numpy as np
from os import makedirs
from os.path import basename, splitext

###########
## INPUT ##
###########

df_fn = sys.argv[1]
df = pd.read_csv(df_fn, sep="\t", index_col=0)

umls_df = pd.read_csv("data/UMLS/MRCONSO.RRF", sep="|", header=None, index_col=False,
                      usecols=[0,2,4,6,10,11], dtype=str)
umls_df.columns = ["CUI","TS", "STT", "ISPREF", "SDUI", "SAB"]
umls_df = umls_df.loc[umls_df["SAB"]=="MDR"]

meddra_hier = pd.read_csv("data/MedDRA/mdhier.asc", sep="$", header=None, index_col=False,
                          usecols=[0,1,2], dtype=str)
meddra_hier.columns = ["PT", "HLT", "HLGT"]

#############
## PROCESS ##
#############

n_unmapped = 0
n_hier_unmapped = 0
out_dicts = {"PT":{}, "HLT":{}, "HLGT": {}}
for i, cui in enumerate(df.columns):
    # To MedDRA IDs
    umls_df_cui = umls_df.loc[(umls_df["CUI"]==cui) & (umls_df["TS"]=="P")
                              & (umls_df["STT"]=="PF") & (umls_df["ISPREF"]=="Y")]
    if umls_df_cui.shape[0] != 1:
        assert umls_df_cui.shape[0] == 0
        n_unmapped += 1
        continue
    meddra_id = umls_df_cui.iloc[0,:]["SDUI"]

    # To MedDRA hierarchies    
    meddra_hier_cui = meddra_hier.loc[meddra_hier["PT"]==meddra_id]
    if meddra_hier_cui.shape[0] == 0:
        n_hier_unmapped += 1
        continue
    meddra_hier_cui = meddra_hier_cui.values.tolist()
    
    # Convert side effect matrix        
    for j, level in enumerate(["PT", "HLT", "HLGT"]):
        for hier in meddra_hier_cui:
            if hier[j] not in out_dicts[level]:
                out_dicts[level][hier[j]] = np.array(df.loc[:,cui])
            else:
                out_dicts[level][hier[j]] |= np.array(df.loc[:,cui])

print("\n{}/{} unmapped".format(n_unmapped, i+1))
print("\n{} hierarchy not found".format(n_hier_unmapped))
            
############
## OUTPUT ##
############

df_basename = splitext(basename(df_fn))[0]
makedirs("data/SIDER/{}".format(df_basename), exist_ok=True)
for level in ["PT", "HLT", "HLGT"]:
    out_df = pd.DataFrame(out_dicts[level])
    out_df.index = df.index
    out_df.to_csv("data/SIDER/{}/MedDRA_{}.tsv".format(df_basename, level), sep="\t")
