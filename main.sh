#!/bin/bash

#############
## SCRIPTS ##
#############

# Prepare data
python3 scripts/all_se_to_matrix.py
python3 scripts/umls_matrix_to_meddra.py data/SIDER/cid_umls_matrix.tsv

# Extra steps to prepare data for CS
python3 scripts/ADR_similarity_labels_to_CUI.py
python3 scripts/similarity_matrix_to_meddra.py data/CS/ADR_PATH_LESK_MATRIX_CUI

# Run models
python3 scripts/get_CV_predictions.py PT data/CS/FEATURE_VECTORS data/SIDER/cid_umls_matrix/MedDRA_PT.tsv
python3 scripts/get_CV_predictions.py HLT data/CS/FEATURE_VECTORS data/SIDER/cid_umls_matrix/MedDRA_HLT.tsv
python3 scripts/get_CV_predictions.py HLGT data/CS/FEATURE_VECTORS data/SIDER/cid_umls_matrix/MedDRA_HLGT.tsv

##############
## Analysis ##
##############

# Theoretical
cd analysis
Rscript beta_approx_intuition.R
Rscript fit_beta_to_Y.R
cd ..
python3 analysis/auc_aupr_sim.py
python3 analysis/estimate_baseline_aupr.py

# Compute metrics
python3 analysis/compute_metrics.py

# Micro-averaged analysis
cd analysis
Rscript micro_metrics.R

# Samplewise analysis
Rscript samplewise_boxplot.R

# Labelwise analysis
Rscript labelwise_boxplot.R
Rscript labelwise_AUC_cor.R
Rscript analyze_significant_labels.R
Rscript auc_sig_by_se_freq.R
cd ..
