#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code evaluates logistig regression with weak labels.
    It compares different algorithms for different datasets.

    Author: JCS, Oct, 2016
"""

# External modules
import numpy as np
import pandas as pd
import sklearn.datasets as skd
# import sklearn.linear_model as sklm
import sklearn.model_selection as skms
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import time
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
n_sim = 10      # No. of simulation runs to average

# Parameters of the classiffier fit method
rho = float(1)/5000    # Learning step

# Parameters of the weak label model
alpha = 0.8
beta = 0.5
gamma = 0.2

# Optimization algorithm to be used for all classifiers
optim = 'GD'     # Options: GD, BFGS

# Cross-validation technique for hyperparameter optimization
# E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
inner_cv = skms.KFold(n_splits=5)

# Non_nested parameter search and scoring
out_path = '../output/'

# Dataset list
# datasetNames = ['blobs', 'blobs2', 'iris']
datasetNames = ['blobs', 'blobs2', 'iris', 'digits']

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
Pe_tr = {}
Pe_cv = {}
Pe_tr_mean = {}
Pe_cv_mean = {}
Pe_cv_std = {}
Pe_cv_pair = {}
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


# #################################
# Running and evaluating algorithms
for dname in datasetNames:

    # Load dataset
    X, y = getDataset(dname)
    n_classes = len(np.unique(y))
    ns, nf = X.shape           # Sample size and data dimension
    X = StandardScaler().fit_transform(X)      # Normalize data

    # ## Report data used in the simulation
    print '--- Dataset: ' + dname
    print '        Sample size: n = {0}'.format(ns)
    print '        Data dimension = {0}'.format(nf)

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

    Pe_tr[dname] = {}
    Pe_cv[dname] = {}
    Pe_tr_mean[dname] = {}
    Pe_cv_mean[dname] = {}
    Pe_cv_std[dname] = {}
    Pe_cv_pair[dname] = {}

    # Parameter values for the logistic regression
    n_it = 4*ns
    if optim == 'GD':
        params = {'rho': rho, 'n_it': n_it}
    else:
        params = {'alpha': 100}

    for tag in tag_list:

        wLR[tag] = wlc.WeakLogisticRegression(
            n_classes, method=method[tag], optimizer=optim, params=params)

        if p_grid[tag] is None:
            clf = GridSearchCV(
                estimator=wLR[tag], param_grid=p_grid['tag'], cv=inner_cv)
        else:
            clf = wLR[tag]

            Pe_tr[dname][tag], Pe_cv[dname][tag] = evaluateClassif(
                clf, X, y, target[labeltype[tag]], n_sim=n_sim)

            # Pe_tr[tag], Pe_cv[tag] = evaluateClassif(
            #     wLR[tag], X, y, target[tag], n_sim=n_sim)
            Pe_tr_mean[dname][tag] = np.mean(Pe_tr[dname][tag])
            Pe_cv_mean[dname][tag] = np.mean(Pe_cv[dname][tag])
            Pe_cv_std[dname][tag] = np.std(Pe_cv[dname][tag])
            Pe_cv_pair[dname][tag] = (Pe_cv_mean[dname][tag],
                                      Pe_cv_std[dname][tag])

        print dname + ': ' + title[tag]
        print '* Average train error = {0}'.format(Pe_tr_mean[dname][tag])
        print '* Average cv error = {0}'.format(Pe_cv_mean[dname][tag])

# #################
# # ## Plot results

# Config plots.
font = {'family': 'Verdana', 'weight': 'regular', 'size': 10}
matplotlib.rc('font', **font)

# Plot error barplots.
for dname in datasetNames:
    plt.figure()
    plt.title(dname)
    for i, tag in enumerate(tag_list):
        plt.scatter([i + 1]*n_sim, Pe_cv[dname][tag], c=[i]*n_sim, s=20,
                    cmap='copper')
        plt.xticks(range(1, 1 + len(tag_list)), tag_list, rotation='45')
    
    plt.show(block=False)

# ################
# Save to csv file
resultsPeCVmean = pd.DataFrame(Pe_cv_mean)
resultsPeCVmean.to_csv(out_path + 'PeCVMean.csv')
resultsPeCVmean.to_latex(out_path + 'PeCVmean.tex')

resultsPeCVpair = pd.DataFrame(Pe_cv_pair)
resultsPeCVpair.to_csv(out_path + 'PeCVpair.csv')
resultsPeCVpair.to_latex(out_path + 'PeCVpair.tex')

print '================'
print 'Fin de ejecucion'
ipdb.set_trace()


