#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This is a first implementation of a classifier algorithm combining:

    - Weak labels
    - Active learning
    - Importance weighting
    - Label recycling.

    Author: JCS, May. 2016
"""

# External modules
import numpy as np
import sklearn.datasets as skd           # Needs version 0.14 or higher
import sklearn.linear_model as sklm
import sklearn.cross_validation as skcv
import ipdb
import matplotlib.pyplot as plt

# My modules
import activelearning.activelearner as al
import wlc.WLclassifier as wlc
import wlc.WLweakener as wlw
import time


def compute_sample_eff(Pe_ref, Pe_test, nq):
    """
    Compute the sample efficiency of error rate array Pe_test with respect
    to the reference error rate Pe_ref.
    """

    m = np.zeros(len(Pe_ref))

    for k, pk in enumerate(Pe_ref):

        # Find the pool where AL got an error rate of at most pk
        dif = pk - Pe_test
        i1 = np.argmax(np.array(dif) >= 0)

        if i1 == 0 and dif[0] < 0:
            i1 = len(nq)-1
            i2 = len(nq)-1
        elif dif[i1] == 0 or i1 == 0:
            i2 = i1
        else:
            i2 = i1 - 1

        # Transform pool values into their corresponding number of samples
        m1 = nq[i1]
        m2 = nq[i2]

        q1 = Pe_test[i1]
        q2 = Pe_test[i2]

        # Interpolate m at pk, between q1 and q2
        if q2 != q1:
            m[k] = m1 + (pk-q1)/(q2-q1)*(m2-m1)
        else:
            m[k] = m1

    return m


def evaluateClassif(X, y, v=None, n_sim=1):

    # Default v
    if v is None:
        v = y

    # ## Initialize aggregate results
    Pe_tr = [0] * n_sim
    Pe_cv = [0] * n_sim

    print '    Averaging {0} simulations. Estimated time...'.format(n_sim)

    # ## Loop over simulation runs
    for i in xrange(n_sim):

        if i == 0:
            start = time.clock()

        # ########################
        # Ground truth evaluation:
        myClassifier.fit(X, v)
        f = myClassifier.predict_proba(X)

        # Then, we evaluate this classifier with all labels
        # Note that training and test samples are being used in this
        # error rate.
        d = np.argmax(f, axis=1)
        Pe_tr[i] = float(np.count_nonzero(y != d)) / ns

        # ##############
        # Self evaluation.
        # First, we compute leave-one-out predictions
        n_folds = min(10, ns)
        preds = skcv.cross_val_predict(myClassifier, X, v,
                                       cv=n_folds, verbose=0)

        # We estimate error rates with the AL labels, in 2 ways:
        Pe_cv[i] = (float(np.count_nonzero(y != preds)) / ns)

        if i == 0:
            tt = time.clock() - start
            print str(tt*n_sim) + ' segundos'

    return Pe_tr, Pe_cv

###############################################################################
# ## MAIN #####################################################################
###############################################################################

############################
# ## Configurable parameters

# Parameters for sklearn synthetic data
ns = 800    # Sample size
nf = 2      # Data dimension
n_classes = 3 # Number of classes

# Common parameters for all AL algorithms
threshold = 0.5
n_sim = 10      # No. of simulation runs to average

# Parameters of the classiffier fit method
rho = float(1)/1000    # Learning step
n_it = 200   # Number of iterations

# Parameters of the weak label model
alpha = 0.2 
beta = 0.2
gamma = 0.2
method = 'quasi_IPL'

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

# Generate weak labels
M = wlw.computeM(n_classes, alpha=alpha, beta=beta, gamma=gamma,
                 method=method)
z = wlw.generateWeak(y, M, n_classes)
v = wlw.computeVirtual(z, n_classes, method=method)

# If dimension is 2, we draw a scatterplot
if nf == 2:
    # Scatterplot.
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='copper')
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.axis('equal')
    plt.show(block=False)

######################
# ## Select classifier

# Create classifier object
# myClassifier = sklm.LogisticRegression(
#     penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
#     intercept_scaling=1, class_weight=None, random_state=None,
#     solver='liblinear', max_iter=100, multi_class='ovr', verbose=0,
#     warm_start=False, n_jobs=1)
myClassifier = wlc.WeakLogisticRegression(n_classes, rho, n_it)

# ## Report data used in the simulation
print '----------------'
print 'Simulation data:'
print '    Sample size: n = {0}'.format(ns)
print '    Data dimensión = {0}'.format(X.shape[1])

############################################################################
# ## PART II: AL algorithm analysis                                      ###
############################################################################

print '----------------------------'
print 'Weak Label Analysis'

# ###################
# Supervised learning
print 'Evaluating learning from clean labels...'
Pe_tr, Pe_cv = evaluateClassif(X, y, y, n_sim=n_sim)
# Average train error rate
Pe_tr_mean = np.mean(Pe_tr)
# Average CV error rate
Pe_cv_mean = np.mean(Pe_cv)

print 'Average train error = {0}'.format(Pe_tr_mean)
print 'Average cv error = {0}'.format(Pe_cv_mean)

# ###################
# Supervised learning
print 'Evaluating learning from weak labels...'
PeW_tr, PeW_cv = evaluateClassif(X, y, v, n_sim=n_sim)
# Average train error rate
PeW_tr_mean = np.mean(PeW_tr)
# Average CV error rate
PeW_cv_mean = np.mean(PeW_cv)

print 'Average train error = {0}'.format(PeW_tr_mean)
print 'Average cv error = {0}'.format(PeW_cv_mean)



# #################
# # ## Plot results

# # Color codes
# color1 = [0.0/255.0, 0.0/255.0, 0.0/255.0]
# color2 = [0.0/255.0, 152.0/255.0, 195.0/255.0]
# color3 = [177.0/255.0, 209.0/255.0, 55.0/255.0]
# color4 = [103.0/255.0, 184.0/255.0, 69.0/255.0]
# color5 = [8.0/255.0, 128.0/255.0, 127.0/255.0]
# color6 = [46.0/255.0, 50.0/255.0, 110.0/255.0]
# color7 = [134.0/255.0, 37.0/255.0, 98.0/255.0]
# color8 = [200.0/255.0, 16.0/255.0, 58.0/255.0]
# color9 = [194.0/255.0, 53.0/255.0, 114.0/255.0]
# color10 = [85.0/255.0, 53.0/255.0, 123.0/255.0]
# color11 = [15.0/255.0, 100.0/255.0, 170.0/255.0]
# color12 = [68.0/255.0, 192.0/255.0, 193.0/255.0]
# color13 = [27.0/255.0, 140.0/255.0, 76.0/255.0]
# color14 = [224.0/255.0, 208.0/255.0, 63.0/255.0]
# color15 = [226.0/255.0, 158.0/255.0, 47.0/255.0]
# color16 = [232.0/255.0, 68.0/255.0, 37.0/255.0]

# font = {'family': 'Verdana',
#         'weight': 'regular',
#         'size': 10}
# # matplotlib.rc('font', **font)

# # Plot error rates vs labeled samples

# # ## Vector of succesive sample sizes
# nq = range(pool_size, ns+pool_size, pool_size)
# nq[-1] = min(nq[-1], ns)
# n_pools = len(nq)

# print 'Size of the feature matrix = ' + str(X.shape)
# print 'Number of labels = ' + str(ns)

# fig = plt.figure()
# h1, = plt.plot(nq, Pe_random, '--', marker='o', color=color1)
# h2, = plt.plot(nq, Pe_AL, '-', marker='.', color=color2)
# plt.legend([h1, h2], ["Random Sampling", "Active Learning"])
# fig.suptitle(u'Testing algorithms')
# plt.xlabel(u'Labeled dataset size')
# plt.ylabel('Error rate (computed over the whole dataset)')
# plt.show(block=False)

# fig = plt.figure()
# h1, = plt.plot(nq, Pe_AL, '-', marker='.', color=color2)
# h2, = plt.plot(nq, PeRaw_AL, '-', marker='.', color=color3)
# h3, = plt.plot(nq, PeW_AL, '-', marker='.', color=color4)

# # Text in the figures
# plt.legend([h1, h2, h3],
#            ["True error rates", "Raw sampling", "Importance sampling"])
# fig.suptitle(u'Evolución del nº de errores con el nº de datos etiquetados')
# plt.xlabel(u'Labeled dataset size')
# plt.ylabel('Error rate')
# plt.draw()
# # plt.show(block=False)


# ######################
# # ## Sample efficiency
# m = np.zeros(n_pools)
# Pe_AL_opt = np.minimum.accumulate(Pe_AL)
# Pe_AL_pes = np.maximum.accumulate(Pe_AL[::-1])[::-1]

# m_opt = compute_sample_eff(Pe_random, Pe_AL_opt, nq)
# m_pes = compute_sample_eff(Pe_random, Pe_AL_pes, nq)
# m_opt_ext = np.append(0, m_opt)
# m_pes_ext = np.append(0, m_pes)
# nq_ext = np.append(0, nq)

# fig = plt.figure()
# plt.plot(nq_ext, m_pes_ext, '-', marker='.', color=color3)
# plt.fill_between(nq_ext, 0, m_pes_ext, color=color3)
# plt.plot(nq_ext, m_opt_ext, '-', marker='.', color=color2)
# plt.fill_between(nq_ext, 0, m_opt_ext-1, color=color2)
# plt.plot(nq_ext, nq_ext, '-', color=color1)
# plt.axis([0, max(nq_ext), 0, max(nq_ext)])
# fig.suptitle(u'Eficiencia muestral del Active Learning')
# plt.xlabel(u'Demanda de etiquetas sin AL')
# plt.ylabel(u'Demanda de etiquetas con AL')

# plt.show(block=False)

ipdb.set_trace()

print '================'
print 'Fin de ejecucion'
