#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code evaluates logistig regression with weak labels

    Author: JCS, June, 2016
            MPN, April, 2017
"""

# External modules
import warnings

import numpy as np
import sklearn.datasets as skd
# import sklearn.linear_model as sklm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize

# My modules
import wlc.WLclassifier as wlc
import wlc.WLweakener as wlw

import keras_models as km
from testUtils import plot_data, plot_results, evaluateClassif

warnings.filterwarnings("ignore")
np.random.seed(42)

###############################################################################
# ## MAIN #####################################################################
###############################################################################

############################
# ## Configurable parameters

# Parameters for sklearn synthetic data
ns = 400           # Sample size
nf = 2             # Data dimension
n_classes = 10      # Number of classes
problem = 'blobs'  # 'blobs' | 'gauss_quantiles' | 'digits'

# Common parameters for all AL algorithms
n_sim = 10       # No. of simulation runs to average

# Parameters of the classiffier fit method
rho = float(1)/5000    # Learning step
n_it = 2*ns            # Number of iterations

# Parameters of the weak label model
alpha = 0.8
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
if problem == 'blobs':
    X, y = skd.make_blobs(n_samples=ns, n_features=nf, centers=n_classes,
                          cluster_std=1.0, center_box=(-10.0, 10.0),
                          shuffle=True, random_state=None)
elif problem == 'gauss_quantiles':
    X, y = skd.make_gaussian_quantiles(n_samples=ns, n_features=nf,
                                       n_classes=n_classes, shuffle=True,
                                       random_state=None)
elif problem == 'digits':
    X, y = skd.load_digits(n_class=n_classes, return_X_y=True)
    nf = X.shape[0]             # Data dimension
    n_it = 10            # Number of iterations
else:
    raise("Problem type unknown: {}".format(problem))
X = StandardScaler().fit_transform(X)

# Convert y into a binary matrix
y_bin = label_binarize(y, range(n_classes))

# Generate weak labels
M = wlw.computeM(n_classes, alpha=alpha, beta=beta, gamma=gamma, method=method)
z = wlw.generateWeak(y, M, n_classes)
v = wlw.computeVirtual(z, n_classes, method=method)
v2 = wlw.computeVirtual(z, n_classes, method=method2, M=M)

# Convert z to a list of binary lists (this is for the OSL alg)
z_bin = wlw.computeVirtual(z, n_classes, method='IPL')

# If dimension is 2, we draw a scatterplot
if nf >= 2:
    plot_data(X, y)

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
n_jobs = {}
x_dict = {}
y_dict = {}
v_dict = {}
Pe_tr = {}
Pe_cv = {}
Pe_tr_mean = {}
Pe_cv_mean = {}
params = {'rho': rho, 'n_it': n_it}
tag_list = []

# ##########################
# Supervised learning (BFGS)
tag = 'Superv-BFGS'
title[tag] = 'Learning from clean labels with BFGS:'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL',
                                      optimizer='BFGS', params=params)
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = y
tag_list.append(tag)

# ############################################
# Optimistic Superset Learning (OSL) with BFGS
tag = 'OSL-BFGS'
title[tag] = 'Optimistic Superset Loss (OSL) with BFGS'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL',
                                      optimizer='BFGS')
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = z_bin
tag_list.append(tag)

# # ############################################
# # Add hoc M-proper loss with Gradient Descent
tag = 'Mproper-GD'
title[tag] = 'M-proper loss with Gradient Descent'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL', optimizer='GD',
                                      params=params)
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = v2
tag_list.append(tag)

# # ############################################
# # Add hoc M-proper loss with BFGS
tag = 'Mproper-BFGS'
title[tag] = 'M-proper loss with Gradient Descent'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                      optimizer='BFGS')
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = v2
tag_list.append(tag)

# ############################################
# Virtual Label Learning with Gradient Descent
tag = 'VLL-GD'
title[tag] = 'Virtual Label Learning (VLL) with Gradient Descent'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL', optimizer='GD',
                                      params=params)
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = v
tag_list.append(tag)

# ############################################
# Virtual Label Learning with Gradient Descent
tag = 'VLLc-BFGS'
title[tag] = 'CC-VLL with BFGS'
wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                      optimizer='BFGS')
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = z_bin
tag_list.append(tag)

# ############################################
# Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
tag = 'Keras-LR-Superv-SGD'
title[tag] = 'Keras M-proper loss with Stochastic Gradient Descent'
params = {'n_it': n_it}
wLR[tag] = km.KerasWeakLogisticRegression(input_size=X.shape[1],
                                          output_size=n_classes,
                                          optimizer='SGD',
                                          params=params)
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = y_bin
tag_list.append(tag)

# ############################################
# Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
tag = 'Keras-LR-OSL-SGD'
title[tag] = 'Keras OSL loss with Stochastic Gradient Descent'
params = {'n_it': n_it}
wLR[tag] = km.KerasWeakLogisticRegression(input_size=X.shape[1],
                                          output_size=n_classes,
                                          optimizer='SGD',
                                          OSL=True,
                                          params=params)
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = z_bin
tag_list.append(tag)

# ############################################
# Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
tag = 'Keras-LR-QIPL-SGD'
title[tag] = 'Keras Logistic regression QIPL loss with Stochastic GD'
params = {'n_it': n_it}
wLR[tag] = km.KerasWeakLogisticRegression(input_size=X.shape[1],
                                          output_size=n_classes,
                                          optimizer='SGD',
                                          params=params)
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = v
tag_list.append(tag)

# ############################################
# Miquel: Add hoc M-proper loss with Stochastic Gradient Descent
tag = 'Keras-LR-Mproper-SGD'
title[tag] = 'Keras Logistic regression M-proper loss with Stochastic GD'
params = {'n_it': n_it}
wLR[tag] = km.KerasWeakLogisticRegression(input_size=X.shape[1],
                                          output_size=n_classes,
                                          optimizer='SGD',
                                          params=params)
n_jobs[tag] = -1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = v2
tag_list.append(tag)

# ############################################
# Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
tag = 'Keras-MLP-OSL-SGD'
title[tag] = 'Keras MLP OSL loss with Stochastic Gradient Descent'
params = {'n_it': n_it}
wLR[tag] = km.KerasWeakMultilayerPerceptron(input_size=X.shape[1],
                                            output_size=n_classes,
                                            optimizer='SGD',
                                            OSL=True,
                                            params=params)
n_jobs[tag] = 1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = z_bin
tag_list.append(tag)

# ############################################
# Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
tag = 'Keras-MLP-QIPL-SGD'
title[tag] = 'Keras MLP QIPL loss with Stochastic Gradient Descent'
params = {'n_it': n_it}
wLR[tag] = km.KerasWeakMultilayerPerceptron(input_size=X.shape[1],
                                            output_size=n_classes,
                                            optimizer='SGD',
                                            params=params)
n_jobs[tag] = 1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = v
tag_list.append(tag)

# ############################################
# Miquel: Add hoc M-proper loss with Stochastic Gradient Descent
tag = 'Keras-MLP-Mproper-SGD'
title[tag] = 'Keras MLP M-proper loss with Stochastic Gradient Descent'
params = {'n_it': n_it}
wLR[tag] = km.KerasWeakMultilayerPerceptron(input_size=X.shape[1],
                                            output_size=n_classes,
                                            optimizer='SGD',
                                            params=params)
n_jobs[tag] = 1
x_dict[tag] = X
y_dict[tag] = y
v_dict[tag] = v2
tag_list.append(tag)

# ############
# Evaluation and plot of each model
for i, tag in enumerate(tag_list):
    print tag
    Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], x_dict[tag],
                                             y_dict[tag], v_dict[tag],
                                             n_sim=n_sim, n_jobs=n_jobs[tag])
    plot_results(tag_list[:(i+1)], Pe_tr, Pe_cv, ns, n_classes, n_sim)

# ############
# Print results.
for tag in tag_list:
    Pe_tr_mean[tag] = np.mean(Pe_tr[tag])
    Pe_cv_mean[tag] = np.mean(Pe_cv[tag])

    print title[tag]
    print '* Average train error = {0}'.format(Pe_tr_mean[tag])
    print '* Average cv error = {0}'.format(Pe_cv_mean[tag])

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

print '================'
print 'Fin de ejecucion'
