import numpy as np
from numpy.random import binomial, beta
from random import seed
from sklearn.metrics import average_precision_score

seed(42)
np.random.seed(42)

def sim_aupr(a, b, l=10**6):
    theta = beta(a, b, l)
    y = binomial(1, theta, l)
    return average_precision_score(y, theta)

params = {"PT": (0.36, 14.56),
          "HLT": (0.38, 4.44),
          "HLGT": (0.48, 2.21)}
for level in params:
    print(level, round(sim_aupr(*params[level]), 3))
