"""
Wrap models from the literature into the sklearn API.
"""

import os
import random
import string
import subprocess
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

def add_dim_to_predict_proba(y_pred):
    """
    Converts probabilistic predictions to format expected by sklearn:
        give the probabilities of being negative and positive.
    """
    y_pred = np.array(y_pred)
    assert y_pred.ndim in (1,2)
    
    out = []
    if y_pred.ndim == 1:
        for p in y_pred:
            out.append([1-p, p])
    if y_pred.ndim == 2:
        for r in np.array(y_pred).T:
            out_l = []
            for p in r:
                out_l.append([1-p, p])
            out.append(out_l)
    
    return list(out)

def pred_proba_to_2D(y_pred):
    """
    Simplifies probabilistic predictions from the sklearn format:
        give only the probability of being positive.
    """
    y_pred_2D = []
    for l in y_pred:
        y_pred_2D.append([li[1] if len(li)>1 else 0 for li in l])
    return pd.DataFrame(y_pred_2D).transpose()

###################
## CUSTOM MODELS ##
###################
    
class WrappedEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, model_name, programming_language):
        self.path = "models/" + model_name + "/"
        self.temp_path = self.path + "temp/"
        self.id = ''.join(random.choices(string.ascii_letters + string.digits, k=50))
        self.programming_language = programming_language

    def fit(self, X, y):
        self.classes_ = [np.array([0,1])] * y.shape[1] # may need to be asserted
        os.makedirs(self.temp_path, exist_ok=True)
        X.to_csv(self.temp_path + self.id + "-fit_X.csv", sep="\t")
        pd.DataFrame(y).to_csv(self.temp_path + self.id + "-fit_y.csv", sep="\t")
        if self.programming_language == "R":
            subprocess.run(["Rscript", "fit.R", self.id, *[str(h) for h in self.hyperparameters.values()]], cwd=self.path)
        elif self.programming_language == "MATLAB":
            arg_strs = ["\""+self.id+"\"", *["\""+str(h)+"\"" for h in self.hyperparameters.values()]]
            subprocess.run(["matlab", "-batch", "fit("+",".join(arg_strs)+")"], cwd=self.path)
        return self
        
    def predict_proba(self, X, header=0):
        X.to_csv(self.temp_path + self.id + "-predict_X.csv", sep="\t")
        if self.programming_language == "R":
            subprocess.run(["Rscript", "predict.R", self.id], cwd=self.path)
        elif self.programming_language == "MATLAB":
            subprocess.run(["matlab", "-batch", "predict("+"\""+self.id+"\""+")"], cwd=self.path)
        y = pd.read_csv(self.temp_path + self.id + "-predict_y.csv", sep="\t", header=header)
        return add_dim_to_predict_proba(y)

class DrugClust(WrappedEstimator):
    def __init__(self, n_neighbors=3):
        super().__init__("DrugClust", "R")
        self.n_neighbors = 3
        self.hyperparameters = OrderedDict([("n_neighbors",n_neighbors)])

class FS_MLKNN(WrappedEstimator):

    def __init__(self, n_neighbors=5, feature_selection=True):
        super().__init__("FS-MLKNN", "MATLAB")
        self.n_neighbors = n_neighbors
        self.feature_selection = feature_selection
        self.hyperparameters = OrderedDict([("n_neighbors",n_neighbors), ("feature_selection",feature_selection)])

    def predict_proba(self, X):
        return super().predict_proba(X, header=None)
    
class CS(WrappedEstimator):

    def __init__(self, similarity_matrices_path,
                 J=5, rnk=100, n_iter=100, lR=1.0, lM=0.1, lN=10):
        super().__init__("CS", "MATLAB")
        self.similarity_matrices_path = similarity_matrices_path
        self.J = J
        self.rnk = rnk
        self.n_iter = n_iter
        self.lR = lR
        self.lM = lM
        self.lN = lN
        self.hyperparameters = OrderedDict([("J",J), ("rnk",rnk), ("iter",n_iter), ("lR",lR), ("lM",lM), ("lN", lN)])

    def fit(self, X, y, meddra_level=None, adr_ids=None):
        os.makedirs(self.temp_path, exist_ok=True)
        X.to_csv(self.temp_path + self.id + "-fit_X.csv", sep="\t")
        pd.DataFrame(y).to_csv(self.temp_path + self.id + "-fit_y.csv", sep="\t")
        
        sim_df = pd.read_csv(self.similarity_matrices_path + meddra_level + ".tsv",
                             sep="\t", index_col=0)
        sim_df.index = sim_df.index.map(str)
        sim_df = sim_df.loc[adr_ids, adr_ids]
        sim_df.to_csv(self.temp_path + self.id + "-ADR_similarity.csv", sep="\t")
        
        return self

    def predict_proba(self, X):
        X.to_csv(self.temp_path + self.id + "-predict_X.csv", sep="\t")
        arg_strs = ["\""+self.id+"\"", *["\""+str(h)+"\"" for h in self.hyperparameters.values()]]
        subprocess.run(["matlab", "-batch", "predict("+",".join(arg_strs)+")"], cwd=self.path)
        y = pd.read_csv(self.temp_path + self.id + "-predict_y.csv", sep="\t", header=None, index_col=False)
        return add_dim_to_predict_proba(y)
