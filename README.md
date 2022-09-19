# se_benchmark

This repository contains code that implements and benchmarks side effect prediction models as used in the article ["Evaluating molecular fingerprint-based models of drug side effects against a statistical control"](https://doi.org/10.1016/j.drudis.2022.103364).

Software dependencies include Python, R, and MATLAB, as well as libraries for each one. The required libraries are specified in the Python and R code. The Statistics and Machine Learning Toolbox is required for MATLAB. ```Rscript``` and ```matlab``` must be in your PATH; ```python3``` is used in [main.sh](main.sh) but this script can be easily modified to use ```python``` instead. Our code ran successfully with Python 3.8.10, R 4.2.0, and MATLAB R2022a.

## How to reproduce our results
Download the input data and execute step-by-step the commands provided in [main.sh](main.sh). The input data directory must be placed at the top of the repository (along with e.g. the ```analysis``` and ```scripts``` directories), and should have this structure:

```
data/
├── CS
│   ├── ADR_PATH_LESK_MATRIX
|   ├── FEATURE_VECTORS
│   └── INTERACTION_MATRIX
├── MedDRA
│   └── mdhier.asc
├── SIDER
│   ├── meddra_all_se.tsv
│   └── meddra.tsv
└── UMLS
    └── MRCONSO.RRF
```

To obtain these files, see [CS](https://github.com/poleksic/side-effects), [MedDRA](https://www.meddra.org/), [SIDER](http://sideeffects.embl.de/), and [UMLS](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html).

## How to use the models
All models are either implemented in or wrapped into the [sklearn API](https://scikit-learn.org/stable/index.html). Generic machine learning models are provided directly by sklearn. The code for custom models created by side effect prediction researchers (see References below) can be found in [models](models). We have sought to directly use the code the authors provided alongside their publications whenever possible but have made modifications so that they can be wrapped in sklearn. The sklearn-wrapped versions of these models, which can be imported in Python, can be found in [scripts/models.py](scripts/models.py).

Examples of how both generic sklearn models and custom models can be imported can be found at the beginning of [scripts/model_params.py](scripts/model_params.py), e.g.
```python
from sklearn.neighbors import KNeighborsClassifier
```
and
```python
from models import DrugClust, FS_MLKNN, CS
```

The rest of this script has examples of initializing these models, e.g.
```python
FS_MLKNN(feature_selection=False)
```

[scripts/get_CV_predictions.py](scripts/get_CV_predictions.py) demonstrates how the sklearn API can be used for prediction. To fit a model, call its `fit` function on a matrix of molecular fingerprints (each row a drug, each column a chemical feature) and corresponding matrix of side effects (each row a drug, each column a side effect). The CS model is slightly more complicated in that it requires the names of the side effects and a side effect similarity matrix. For outputting the predictions of a model, simply call the model's `predict_proba` function on a matrix of molecular fingerprints.

## References
**DrugClust**: Dimitri, Giovanna Maria, and Pietro Lió. "DrugClust: a machine learning approach for drugs side effects prediction." *Computational Biology and Chemistry* 68 (2017): 204-210.

**CS**: Poleksic, Aleksandar, and Lei Xie. "Predicting serious rare adverse reactions of novel chemicals." *Bioinformatics* 34.16 (2018): 2835-2842.

**FS-MLKNN**: Zhang, Wen, et al. "Predicting drug side effects by multi-label learning and ensemble learning." *BMC Bioinformatics* 16.1 (2015): 1-11.
