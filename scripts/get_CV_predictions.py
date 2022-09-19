"""
Generate cross-validation predictions of the models specified in model_params.py.
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import KFold
from models import pred_proba_to_2D
from model_params import baseline_models, generic_models, custom_models

results_id = sys.argv[1]
X_fn = sys.argv[2]
y_fn = sys.argv[3]
overwrite = sys.argv[4]=="True" if len(sys.argv) > 4 else False

###################
## SET UP MODELS ##
###################

models = baseline_models + generic_models + custom_models

results_dir = "results/{}/".format(results_id)
os.makedirs(results_dir, exist_ok=True)

with open(results_dir + "model-params.txt", "w") as f:
    for model_name, model in models:
        f.write(model_name + "\t" + str(model) + "\n")

################
## READ INPUT ##
################

X = pd.read_csv(X_fn, sep=" ", index_col=0)
y = pd.read_csv(y_fn, sep="\t", index_col=0)
X.index = y.index

###################
## PREPROCESSING ##
###################

n_cols_pre = y.shape[1]
y = y.loc[:,y.sum(axis=0) >= 15]
print("{}/{} ({}%) SE retained with threshold=15".format(y.shape[1], n_cols_pre,
                                                         round(y.shape[1]/n_cols_pre*100)))
kf = KFold(5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    assert all(y.iloc[train_index,:].sum(axis=0) > 0)
    n_no_drugs = sum(y.iloc[test_index,:].sum(axis=0) == 0)
    print("# SE with no drugs in test fold {}: {}".format(i+1, n_no_drugs))

######################
## CROSS-VALIDATION ##
######################

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print("\nFOLD {}".format(i+1))
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]

    for model_name, model in models:
        output_dir = results_dir + "CV/fold{:02}/".format(i+1)
        out_fn = output_dir + model_name + ".tsv"
        if os.path.isfile(out_fn) and not overwrite:
            print(out_fn, "exists, skipping")
            continue

        print(model_name)
        if model_name == "CS":
            model.fit(X_train, y_train, results_id, y.columns)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_pred = pred_proba_to_2D(y_pred)
        y_pred.index = y_test.index
        y_pred.columns = y_test.columns
        
        os.makedirs(output_dir, exist_ok=True)
        y_pred.to_csv(out_fn, sep="\t")
