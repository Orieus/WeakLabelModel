#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code evaluates logistig regression with weak labels

    Author: JCS, June, 2016
"""

# External modules
import numpy as np
import sklearn.datasets as skd
# import sklearn.linear_model as sklm
import sklearn.cross_validation as skcv
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import time
import ipdb

# My modules
import wlc.WLclassifier as wlc
import wlc.WLweakener as wlw


def evaluateClassif(classif, X, y, v=None, n_sim=1):

    # Default v
    if v is None:
        v = y

    # ## Initialize aggregate results
    Pe_tr = [0] * n_sim
    Pe_cv = [0] * n_sim

    print '    Averaging {0} simulations. Estimated time...'.format(n_sim),

    # ## Loop over simulation runs
    for i in xrange(n_sim):

        if i == 0:
            start = time.clock()

        # ########################
        # Ground truth evaluation:
        classif.fit(X, v)
        f = classif.predict_proba(X)

        # Then, we evaluate this classifier with all labels
        # Note that training and test samples are being used in this
        # error rate.
        d = np.argmax(f, axis=1)
        Pe_tr[i] = float(np.count_nonzero(y != d)) / ns

        # ##############
        # Self evaluation.
        # First, we compute leave-one-out predictions
        n_folds = min(10, ns)
        preds = skcv.cross_val_predict(classif, X, v,
                                       cv=n_folds, verbose=0)

        # Estimate error rates:
        Pe_cv[i] = float(np.count_nonzero(y != preds)) / ns

        if i == 0:
            print '{0} segundos'.format((time.clock() - start) * n_sim)

    return Pe_tr, Pe_cv

###############################################################################
# ## MAIN #####################################################################
###############################################################################

############################
# ## Configurable parameters

# Parameters for sklearn synthetic data
ns = 400        # Sample size
nf = 2         # Data dimension
n_classes = 5   # Number of classes

# Common parameters for all AL algorithms
n_sim = 10      # No. of simulation runs to average

# Parameters of the classiffier fit method
rho = float(1)/5000    # Learning step
n_it = 2*ns           # Number of iterations

# Parameters of the weak label model
alpha = 0.2
beta = 0.2
gamma = 0.2
method = 'quasi_IPL'
method2 = 'Mproper'
# method = 'quasi_IPL_old'

#####################
# ## A title to start

print "======================================"
print "    Testing Learning from Weak Labels "
print "======================================"

###############################################################################
# ## PART I: Load data (samples and true labels)                             ##
###############################################################################

# X, y = skd.make_classification(
#     n_samples=ns, n_features=nf, n_informative=3, n_redundant=0,
#     n_repeated=0, n_classes=n_classes, n_clusters_per_class=2, weights=None,
#     flip_y=0.0001, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
#     shuffle=True, random_state=None)
X, y = skd.make_blobs(
    n_samples=ns, n_features=nf, centers=n_classes, cluster_std=1.0,
    center_box=(-10.0, 10.0), shuffle=True, random_state=None)
X = StandardScaler().fit_transform(X)

# Generate weak labels
M = wlw.computeM(n_classes, alpha=alpha, beta=beta, gamma=gamma,
                 method=method)
z = wlw.generateWeak(y, M, n_classes)
v = wlw.computeVirtual(z, n_classes, method=method)
v2 = wlw.computeVirtual(z, n_classes, method=method2, M=M)

# Convert z to a list of binary lists (this is for the OSL alg)
z_bin = wlw.computeVirtual(z, n_classes, method='IPL')

# If dimension is 2, we draw a scatterplot
if nf == 2:
    # Scatterplot.
    plt.scatter(X[:, 0], X[:, 1], c=2*y, s=20, cmap='copper')
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.title('Labeled dataset')
    plt.axis('equal')
    plt.show(block=False)

######################
# ## Select classifier

# ## Report data used in the simulation
print '----------------'
print 'Simulation data:'
print '    Sample size: n = {0}'.format(ns)
print '    Data dimension = {0}'.format(X.shape[1])

############################################################################
# ## PART II: AL algorithm analysis                                      ###
############################################################################

print '----------------------------'
print 'Weak Label Analysis'

wLR = {}
title = {}
Pe_tr = {}
Pe_cv = {}
Pe_tr_mean = {}
Pe_cv_mean = {}
params = {'rho': rho, 'n_it': n_it}
tag_list = []

# ###################
# Supervised learning
tag = 'Supervised'
tag_list.append(tag)
title[tag] = 'Learning from clean labels:'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL', optimizer='GD',
                                      params=params)
Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, y, n_sim=n_sim)

# # ##########################
# # Supervised learning (BFGS)
# tag = 'Superv-BFGS'
# tag_list.append(tag)
# title[tag] = 'Learning from clean labels with BFGS:'
# wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL',
#                                       optimizer='BFGS', params=params)
# Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, y, n_sim=n_sim)

# # ##################################
# # Optimistic Superset Learning (OSL)
# tag = 'OSL'
# tag_list.append(tag)
# title[tag] = 'Optimistic Superset Loss (OSL)'
# wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL', optimizer='GD',
#                                       params=params)
# Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, z_bin, n_sim=n_sim)

# ############################################
# Optimistic Superset Learning (OSL) with BFGS
tag = 'OSL-BFGS'
tag_list.append(tag)
title[tag] = 'Optimistic Superset Loss (OSL) with BFGS'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL',
                                      optimizer='BFGS')
Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, z_bin, n_sim=n_sim)

# # ############################################
# # Virtual Label Learning with Gradient Descent
tag = 'Mproper-GD'
tag_list.append(tag)
title[tag] = 'M-proper loss with Gradient Descent'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL', optimizer='GD',
                                       params=params)
Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, v2, n_sim=n_sim)

# ############################################
# Virtual Label Learning with Gradient Descent
tag = 'VLL-GD'
tag_list.append(tag)
title[tag] = 'Virtual Label Learning (VLL) with Gradient Descent'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL', optimizer='GD',
                                      params=params)
Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, v, n_sim=n_sim)

# ###################################################
# Virtual Label Learning with BFGS and regularization
tag = 'VLL-BFGS'
tag_list.append(tag)
title[tag] = 'Virtual Label Learning (VLL) with BFGS and regularization'
params = {'alpha': (2.0 + nf)/2}    # This value for alpha is an heuristic
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                      optimizer='BFGS', params=params)
Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, v, n_sim=n_sim)

# ############################################
# Virtual Label Learning with Gradient Descent
tag = 'VLLc-GD'
tag_list.append(tag)
title[tag] = 'CC-VLL with Gradient Descent'
params = {'rho': rho, 'n_it': n_it}
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL', optimizer='GD',
                                      params=params)
Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, z_bin, n_sim=n_sim)

# ############################################
# Virtual Label Learning with Gradient Descent
tag = 'VLLc-BFGS'
tag_list.append(tag)
title[tag] = 'CC-VLL with BFGS'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                      optimizer='BFGS')
Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y, z_bin, n_sim=n_sim)

# ############
# Print results.
for tag in tag_list:
    Pe_tr_mean[tag] = np.mean(Pe_tr[tag])
    Pe_cv_mean[tag] = np.mean(Pe_cv[tag])

    print title[tag]
    print '* Average train error = {0}'.format(Pe_tr_mean[tag])
    print '* Average cv error = {0}'.format(Pe_cv_mean[tag])

# #################
# # ## Plot results

# Config plots.
font = {'family': 'Verdana', 'weight': 'regular', 'size': 10}
matplotlib.rc('font', **font)

# Plot error barplots.
plt.figure()
for i, tag in enumerate(tag_list):
    plt.scatter([i + 1]*n_sim, Pe_cv[tag], c=[i]*n_sim, s=20, cmap='copper')

plt.xticks(range(1, 1 + len(tag_list)), tag_list, rotation='45')
plt.show(block=False)

# Plot decision boundaries.
# Plotting decision regions
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))

# f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

# for idx, clf, tt in zip(product([0, 1], [0, 1]),
#                         [clf1, clf2, clf3, eclf],
#                         ['Decision Tree (depth=4)', 'KNN (k=7)',
#                          'Kernel SVM', 'Soft Voting']):

#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
#     axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
#     axarr[idx[0], idx[1]].set_title(tt)

# plt.show()

ipdb.set_trace()

print '================'
print 'Fin de ejecucion'
