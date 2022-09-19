import sys
import pandas as pd
from os import makedirs

df_fn = sys.argv[1]
df = pd.read_csv(df_fn, sep="\t", index_col=0)

###################
## TO MEDDRA IDS ##
###################

umls_df = pd.read_csv("data/UMLS/MRCONSO.RRF", sep="|", header=None, index_col=False,
                      usecols=[0,2,4,6,10,11], dtype=str)
umls_df.columns = ["CUI","TS", "STT", "ISPREF", "SDUI", "SAB"]
umls_df = umls_df.loc[umls_df["SAB"]=="MDR"]

meddra_hier = pd.read_csv("data/MedDRA/mdhier.asc", sep="$", header=None, index_col=False,
                          usecols=[0,1,2], dtype=str)
meddra_hier.columns = ["PT", "HLT", "HLGT"]

cui_to_meddra = {}
cui_to_hiers = {}
n_unmapped = 0
n_hier_unmapped = 0
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
    cui_to_hiers[cui] = meddra_hier_cui.values.tolist()

print("\n{}/{} unmapped".format(n_unmapped, i+1))
print("\n{} hierarchy not found".format(n_hier_unmapped))

#############
## CONVERT ##
#############

makedirs("data/CS/ADR_similarity", exist_ok=True)

out_dfs = {}
meddra_to_cuis = {"PT":{}, "HLT":{}, "HLGT":{}}
for j, level in enumerate(["PT", "HLT", "HLGT"]):
    print(level)
    
    # For each MedDRA ID at each level, record which CUIs map to it
    for cui in cui_to_hiers:
        for hier in cui_to_hiers[cui]:
            meddra_id = hier[j]
            if meddra_id in meddra_to_cuis[level]:
                meddra_to_cuis[level][meddra_id].add(cui)
            else:
                meddra_to_cuis[level][meddra_id] = set([cui])
    
    # Recompute the similarity matrix at higher MedDRA levels by averaging its
    # CUI similarities
    
    # Along rows
    mean1_list = []
    for i, meddra_id in enumerate(meddra_to_cuis[level]):
        cuis = meddra_to_cuis[level][meddra_id]
        mean1 = df.loc[cuis,:].mean(axis=0)
        mean1_list.append(mean1)
    mean1_df = pd.DataFrame(mean1_list,
                            index=meddra_to_cuis[level].keys(),
                            columns=df.columns)
    
    # Along columns
    mean2_list_t = []
    for i, meddra_id in enumerate(meddra_to_cuis[level]):
        cuis = meddra_to_cuis[level][meddra_id]
        mean2 = mean1_df.loc[:,cuis].mean(axis=1)
        mean2_list_t.append(mean2)
    out_df = pd.DataFrame(mean2_list_t).T
    out_df.index = meddra_to_cuis[level].keys()
    out_df.columns = meddra_to_cuis[level].keys()

    out_df.to_csv("data/CS/ADR_similarity/{}.tsv".format(level), sep="\t")
