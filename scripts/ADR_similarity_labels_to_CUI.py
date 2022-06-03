import pandas as pd

sim_df = pd.read_csv("data/CS/ADR_PATH_LESK_MATRIX",
                     sep=" ", skiprows=[0], header=None, index_col=False)
sim_df.drop(sim_df.columns[-1], axis=1, inplace=True)
with open("data/CS/INTERACTION_MATRIX") as f:
    cui_header = f.readline().rstrip().split(" ")[1:]
sim_df.columns = cui_header
sim_df.index = cui_header
sim_df.to_csv("data/CS/ADR_PATH_LESK_MATRIX_CUI", sep="\t")
