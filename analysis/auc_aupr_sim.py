""" Simulates surfaces of AUC and AUPR with respect to parameters of Beta. """

import numpy as np
from numpy.random import binomial, beta
from random import seed
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

seed(42)
np.random.seed(42)

# Simulation sizes
l = 10**4 # Number of labels
N_approx = 10**5 # Number of samples for computing approximation

# Find alpha and beta for given mean and variance
def params(mean, var):
    assert var < mean*(1-mean)
    alpha = mean * ( (mean*(1-mean))/var - 1 )
    beta = (1-mean) * ( (mean*(1-mean))/var - 1 )
    return alpha, beta

# Set up plotting matrices
N_means = 50
N_variances = 50
means = np.linspace(0.02, 0.98, N_means)
variances = np.linspace(0.0001, 0.019, N_variances)
X, Y = np.meshgrid(means, variances)
Z_auc = np.zeros((N_variances, N_means)) # AUC
Z_approx_diff = np.zeros((N_variances, N_means)) # Difference between AUC and approximation
Z_aupr = np.zeros((N_variances, N_means)) # AUPR

# Simulate
for i in range(len(X[0])):
    for j in range(len(X)):
        a, b = params(means[i], variances[j])
        theta = beta(a, b, l)
        Z_approx = np.mean(beta(a+1,b, N_approx) > beta(a,b+1, N_approx))
        
        y = binomial(1, theta, l)
        Z_auc[j,i] = roc_auc_score(y, theta)
        Z_approx_diff[j,i] = Z_approx - Z_auc[j,i]
        Z_aupr[j,i] = average_precision_score(y, theta)

## Plot AUC ##

ax = plt.axes(projection="3d")
ax.plot_surface(X, np.sqrt(Y), Z_auc,
                rstride=1, cstride=1, cmap="viridis", edgecolor="none")
ax.set_xlim(0, 1)
ax.set_ylim(0, max(np.sqrt(variances)))
ax.set_zlim(Z_auc.flatten().min(), 1)
ax.azim = 55
ax.set_xlabel("Mean")
ax.set_ylabel("Standard deviation")
ax.set_zlabel("AUC", rotation=90)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.figure.tight_layout()
ax.figure.savefig("analysis/figures/surface_AUC.pdf", bbox_inches="tight")
plt.clf()

CS = plt.contour(X, np.sqrt(Y), Z_auc)
plt.clabel(CS, CS.levels, inline=True, fmt="%.2f")
plt.xlim(0, 1)
plt.ylim(0, max(np.sqrt(variances)))
plt.xlabel("Mean")
plt.ylabel("Standard deviation")
plt.tight_layout()
plt.savefig("analysis/figures/contour_AUC.pdf")
plt.clf()

## Plot AUC approximation difference ##

plt.hist(Z_approx_diff.flatten(), color="grey", bins=100)
plt.xlabel("$\mathrm{AUC}_\mathrm{approx} - \mathrm{AUC}$")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("analysis/figures/approx_diff_AUC_hist.pdf")
plt.clf()

## Plot AUPR ##

ax = plt.axes(projection="3d")
ax.plot_surface(X, np.sqrt(Y), Z_aupr,
                rstride=1, cstride=1, cmap="viridis", edgecolor="none")
ax.set_xlim(0, 1)
ax.set_ylim(0, max(np.sqrt(variances)))
ax.set_zlim(Z_aupr.flatten().min(), 1)
ax.azim = 60
ax.elev = 33
ax.set_xlabel("Mean")
ax.set_ylabel("Standard deviation")
ax.set_zlabel("AUPR", rotation=90)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.figure.tight_layout()
ax.figure.savefig("analysis/figures/surface_AUPR.pdf", bbox_inches="tight")
plt.clf()

CS = plt.contour(X, np.sqrt(Y), Z_aupr)
plt.clabel(CS, CS.levels, inline=True, fmt="%.2f")
plt.xlim(0, 1)
plt.ylim(0, max(np.sqrt(variances)))
plt.xlabel("Mean")
plt.ylabel("Standard deviation")
plt.tight_layout()
plt.savefig("analysis/figures/contour_AUPR.pdf")
plt.clf()
