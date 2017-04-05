#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code evaluates logistic regression with weak labels
    Author: JCS, June, 2016
"""

# External modules
import numpy as np
import sklearn.model_selection as skms
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import ipdb

# My modules
import wlc.WLclassifier as wlc
import wlc.WLweakener as wlw
from wlc.WLanalyzer import evaluateClassif
from dataloader.dataLoad import getDataset


###############################################################################
# ## MAIN #####################################################################
###############################################################################

############################
# ## Configurable parameters

# Common parameters for all AL algorithms
n_sim = 20      # No. of simulation runs to average

# Parameters of the classiffier fit method
rho = float(1)/5000    # Learning step

# Parameters of the weak label model
alpha = 0.8
beta = 0.2
gamma = 0.2

# Optimization algorithm to be used for all classifiers
optim = 'BFGS'     # Options: GD, BFGS

# Cross-validation technique for hyperparameter optimization
# E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
inner_cv = skms.KFold(n_splits=5)

# Non_nested parameter search and scoring
out_path = '../output/'

# Dataset list
# datasetName = ['blobs', 'blobs2', 'iris']
datasetName = 'blobs'

#####################
# ## A title to start

print "======================================"
print "    Testing Learning from Weak Labels "
print "======================================"

############################################################################
# ## Algorithm analysis                                                  ###
############################################################################

print '----------------------------'
print 'Weak Label Analysis'

wLR = {}
title = {}
method = {}
labeltype = {}
p_grid = {}
tag_list = []

# ########################
# Dictionary of algorithms

grid = {'rho': [rho*r for r in[1, 10]]}

# Supervised learning
tag = 'Supervised'
tag_list.append(tag)
title[tag] = 'Learning from clean labels:'
labeltype[tag] = 'true'
method[tag] = 'OSL'
p_grid[tag] = grid

# Optimistic Superset Learning (OSL)
tag = 'OSL'
tag_list.append(tag)
title[tag] = 'Optimistic Superset Loss (OSL)'
labeltype[tag] = 'weak'
method[tag] = 'OSL'
p_grid[tag] = grid

# Add hoc M-proper loss with Gradient Descent
tag = 'Mproper-GD'
tag_list.append(tag)
title[tag] = 'M-proper loss'
labeltype[tag] = 'virtM'
method[tag] = 'VLL'
p_grid[tag] = grid

# Virtual Label Learning with Gradient Descent
tag = 'VLL-GD'
tag_list.append(tag)
title[tag] = 'Virtual Label Learning (VLL)'
labeltype[tag] = 'virt'
method[tag] = 'VLL'
p_grid[tag] = grid

# Virtual Label Learning with Gradient Descent
tag = 'VLLc-GD'
tag_list.append(tag)
title[tag] = 'Regularized VLL'
labeltype[tag] = 'weak'
method[tag] = 'VLL'
p_grid[tag] = grid

# Load dataset
X, y = getDataset(datasetName)
n_classes = len(np.unique(y))
ns, nf = X.shape           # Sample size and data dimension
X = StandardScaler().fit_transform(X)      # Normalize data

# ## Report data used in the simulation
print '--- Dataset: ' + datasetName
print '        Sample size: n = {0}'.format(ns)
print '        Data dimension = {0}'.format(nf)

# #################################
# Running and evaluating algorithms

n_p = 8   # Number of values of the parameter to explore
pvalues = np.linspace(0, 0.9, n_p)

Pe_tr = {t: np.zeros(n_p) for t in tag_list}
Pe_cv = {t: np.zeros(n_p) for t in tag_list}
Pe_tr_mean = {t: np.zeros(n_p) for t in tag_list}
Pe_cv_mean = {t: np.zeros(n_p) for t in tag_list}
Pe_cv_std = {t: np.zeros(n_p) for t in tag_list}

for k, p in enumerate(pvalues):

    beta = p

    print beta

    # Generate weak labels
    M = wlw.computeM(n_classes, alpha=alpha, beta=beta, gamma=gamma,
                     method='quasi_IPL')
    z = wlw.generateWeak(y, M, n_classes)
    v = wlw.computeVirtual(z, n_classes, method='quasi_IPL')
    v2 = wlw.computeVirtual(z, n_classes, method='Mproper', M=M)

    # Convert z to a list of binary lists (this is for the OSL alg)
    z_bin = wlw.computeVirtual(z, n_classes, method='IPL')

    # Target dictionary
    target = {'true': y,
              'weak': z_bin,
              'virt': v,
              'virtM': v2}

    # Parameter values for the logistic regression
    n_it = 3*ns
    if optim == 'GD':
        params = {'rho': rho, 'n_it': n_it}
    else:
        params = {'alpha': 10}

    for tag in tag_list:

        # Train classifier
        wLR[tag] = wlc.WeakLogisticRegression(
            n_classes, method=method[tag], optimizer=optim, params=params)

        if p_grid[tag] is None:
            clf = GridSearchCV(
                estimator=wLR[tag], param_grid=p_grid['tag'], cv=inner_cv)
        else:
            clf = wLR[tag]

        Pe_tr_all, Pe_cv_all = evaluateClassif(
            clf, X, y, target[labeltype[tag]], n_sim=n_sim)

        Pe_tr_mean[tag][k] = np.mean(Pe_tr_all)
        Pe_cv_mean[tag][k] = np.mean(Pe_cv_all)
        Pe_cv_std[tag][k] = np.std(Pe_cv_all)

# #################
# # ## Plot results

# Config plots.
font = {'family': 'Verdana', 'weight': 'regular', 'size': 10}
matplotlib.rc('font', **font)

# Plot error barplots.
plt.figure()
plt.title('Average error rate')
for i, tag in enumerate(tag_list):
    plt.plot(pvalues, Pe_cv_mean[tag], label=tag)
plt.legend()
plt.xlabel('Parameter of the mixing model')
plt.ylabel('Error rate')
plt.savefig('PeCV' + datasetName +  optim + '.png')
plt.show(block=False)

# Plot error barplots.
plt.figure()
plt.title('Standard deviations')
for i, tag in enumerate(tag_list):
    plt.plot(pvalues, Pe_cv_std[tag], label=tag)
plt.legend()
plt.xlabel('Parameter of the mixing model')
plt.ylabel('Standard deviation')
plt.savefig('StdCV' + datasetName + optim + '.png')
plt.show(block=False)

# ################
# Save to csv file
# resultsPeCVmean = pd.DataFrame(Pe_cv_mean)
# resultsPeCVmean.to_csv(out_path + 'PeCVMean.csv')
# resultsPeCVmean.to_latex(out_path + 'PeCVmean.tex')

# resultsPeCVpair = pd.DataFrame(Pe_cv_pair)
# resultsPeCVpair.to_csv(out_path + 'PeCVpair.csv')
# resultsPeCVpair.to_latex(out_path + 'PeCVpair.tex')

print '================'
print 'Fin de ejecucion'
ipdb.set_trace()


