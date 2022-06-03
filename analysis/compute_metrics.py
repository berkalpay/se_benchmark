import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

metrics_list = []
label_metrics_list = []
sample_metrics_list = []

fig_pr, ax_pr = plt.subplots(5, 3)

for level_i, hier in enumerate(["PT", "HLT", "HLGT"]):
    print("working on {}...".format(hier))

    #######################
    ## LOAD GROUND TRUTH ##
    #######################
    
    y_fn = "data/SIDER/cid_umls_matrix/MedDRA_{}.tsv".format(hier)
    y = pd.read_csv(y_fn, sep="\t", index_col=0)
    y = y.loc[:, y.sum(axis=0) >= 15]
    
    ######################
    ## LOAD PREDICTIONS ##
    ######################
    
    model_names = [s.split(".tsv")[0] for s in os.listdir("results/{}/CV/fold01/".format(hier))]
    y_preds_cv = {mn:{} for mn in model_names}
    y_cv = {}
    for fold in [1,2,3,4,5]:
        dir_path = "results/{}/CV/fold{:02}/".format(hier, fold)
        for j, fn in enumerate(os.listdir(dir_path)):        
            model_name = fn.split(".tsv")[0]
            y_pred_cv_df = pd.read_csv(dir_path + fn, sep="\t", index_col=0)
            y_preds_cv[model_name][str(fold)] = y_pred_cv_df
            if j == 0:
                y_cv[str(fold)] = y.loc[y_pred_cv_df.index,:]
    
    ###############
    ## PR CURVES ##
    ###############

    for fold in [1,2,3,4,5]:
        axc = ax_pr[fold-1, level_i]
        for mn in model_names:
            yt = y_cv[str(fold)]
            yp = y_preds_cv[mn][str(fold)].loc[yt.index,:]
            p, r, _ = precision_recall_curve(yt.values.ravel(), yp.values.ravel())
            axc.step(r, p, where="post", label=mn, linewidth=0.75, alpha=0.75)
        axc.set_ylim([0.0, 1.0])
        axc.set_xlim([0.0, 1.0])
        if hier == "PT":
            axc.set_ylabel("Fold {}".format(fold), rotation=0, labelpad=40)
        else:
            axc.yaxis.set_ticklabels([])
        if fold == 1:
            axc.set_title(hier)
        if fold != 5:
            axc.xaxis.set_ticklabels([])
        if hier == "HLGT" and fold==5:
            axc.legend(fontsize=8, ncol=2, frameon=False)
    
    #####################
    ## COMPUTE METRICS ##
    #####################
    
    # Micro-averaged
    for mn in model_names:
        auc_cv, ap_cv, precision_cv, recall_cv = [], [], [], []
        for fold in y_preds_cv[mn]:
            yt = y_cv[fold]
            yp = y_preds_cv[mn][fold]
            auc_cv.append(roc_auc_score(yt, yp, average="micro"))
            ap_cv.append(average_precision_score(yt, yp, average="micro"))
            yp_bin = (yp > 0.5).astype(int)
            precision_cv.append(precision_score(yt, yp_bin, average="micro"))
            recall_cv.append(recall_score(yt, yp_bin, average="micro"))
        metrics_list.append([mn, hier, np.mean(auc_cv), np.mean(ap_cv), np.mean(precision_cv), np.mean(recall_cv)])
        
    # Labelwise
    j_to_keep = []
    for j in range(y.shape[1]):
        if all(sum(y_cv[str(fold)].iloc[:,j]) for fold in [1,2,3,4,5]):
            j_to_keep.append(j)
    print("{} SE removed to calculate labelwise metrics".format(y.shape[1] - len(j_to_keep)))
    for mn in model_names:
        for j in j_to_keep:
            for fold in y_preds_cv[mn]:
                yt = y_cv[fold].iloc[:,j]
                yp = y_preds_cv[mn][fold].iloc[:,j]
                label_metrics_list.append([fold, y.columns[j], mn, hier,
                                           roc_auc_score(yt, yp),
                                           mannwhitneyu(yp[yt==0], yp[yt==1], alternative="less")[1],
                                           average_precision_score(yt, yp)])
    
    # Samplewise
    drugs_to_keep = set(y.index[y.sum(axis=1) > 0])
    print("{} drugs removed to calculate samplewise metrics".format(sum(y.sum(axis=1) == 0)))
    for mn in model_names:
        for fold in y_preds_cv[mn]:
            ypf = y_preds_cv[mn][fold]
            for drug in ypf.index:
                if drug in drugs_to_keep:
                    yt = y.loc[drug,:]
                    yp = ypf.loc[drug,:]
                    sample_metrics_list.append([drug, mn, hier,
                                                roc_auc_score(yt, yp),
                                                average_precision_score(yt, yp)])
            
####################
## SAVE PR CURVES ##
####################

# From: https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots/36542971#36542971
fig_pr.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Recall")
plt.ylabel("Precision")
fig_pr.set_size_inches(10, 15)
fig_pr.tight_layout()
fig_pr.savefig("analysis/figures/PR_curves.pdf")
            
##################
## SAVE METRICS ##
##################

os.makedirs("analysis/data/", exist_ok=True)

metrics_df = pd.DataFrame(metrics_list, columns=["model", "hier", "AUC", "AUPR", "precision", "recall"])
metrics_df.to_csv("analysis/data/micro_metrics.tsv", index=False, sep="\t")

label_metrics_df = pd.DataFrame(label_metrics_list, columns=["fold", "label", "model", "hier", "AUC", "AUC05_pval", "AUPR"])
label_metrics_df.to_csv("analysis/data/label_metrics.tsv", index=False, sep="\t")

sample_metrics_df = pd.DataFrame(sample_metrics_list, columns=["sample", "model", "hier", "AUC", "AUPR"])
sample_metrics_df.to_csv("analysis/data/sample_metrics.tsv", index=False, sep="\t")
