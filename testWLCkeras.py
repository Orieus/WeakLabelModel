#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code evaluates logistig regression with weak labels

    Author: JCS, June, 2016
            MPN, April, 2017
"""

# External modules
import os
import warnings

import pandas as pd
import numpy as np
import openml
from optparse import OptionParser
import sklearn.datasets as skd
# import sklearn.linear_model as sklm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize

# My modules
import wlc.WLclassifier as wlc
import wlc.WLweakener as wlw

import keras_models as km
from testUtils import plot_data, plot_results, evaluateClassif

from diary import Diary

warnings.filterwarnings("ignore")

# Parameters for sklearn synthetic data
openml_ids = {'iris': 61, 'pendigits': 32, 'glass': 41, 'segment': 36,
              'vehicle': 54, 'vowel': 307, 'wine': 187, 'abalone': 1557,
              'balance-scale': 11, 'car': 21, 'ecoli': 39, 'satimage': 182,
              'collins': 478, 'cardiotocography': 1466, 'JapaneseVowels': 375,
              'autoUniv-au6-1000': 1555, 'autoUniv-au6-750': 1549,
              'analcatdata_dmft': 469, 'autoUniv-au7-1100': 1552,
              'GesturePhaseSegmentationProcessed': 4538,
              'autoUniv-au7-500': 1554, 'mfeat-zernike': 22, 'zoo': 62,
              'page-blocks': 30, 'yeast': 181, 'flags': 285,
              'visualizing_livestock': 685, 'diggle_table_a2': 694,
              'prnn_fglass': 952, 'confidence': 468, 'fl2000': 477}
openml_ids_nans = {'heart-c': 49, 'dermatology': 35}


def parse_arguments():
    parser = OptionParser()
    parser.add_option('-p', '--problems', dest='problems', default='blobs',
                      type=str, help=('List of datasets or toy examples to'
                                      'test separated by with no spaces.'))
    # Parameters for sklearn synthetic data
    parser.add_option('-s', '--n-samples', dest='ns', default=400,
                      type=int, help='Number of samples if toy dataset.')
    parser.add_option('-f', '--n-features', dest='nf', default=2,
                      type=int, help='Number of features if toy dataset.')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=10,
                      type=int, help='Number of classes if toy dataset.')
    # Common parameters for all AL algorithms
    parser.add_option('-m', '--n-simulations', dest='n_sim', default=10,
                      type=int, help='Number of times to run every model.')
    parser.add_option('-l', '--loss', dest='loss', default='square',
                      type=str, help=('Loss function to minimize between '
                                      'square (brier score) or CE (cross '
                                      'entropy)'))
    # Parameters of the classiffier fit method
    parser.add_option('-r', '--rho', dest='rho', default=0.0002,
                      type=float,
                      help='Learning step for the Gradient Descent')
    parser.add_option('-i', '--n-iterations', dest='n_it', default=1000,
                      type=int, help=('Number of iterations of '
                                      'Gradient Descent.'))
    parser.add_option('-e', '--method', dest='method', default='quasi_IPL',
                      type=str, help=('Method to generate the matrix M.'
                                      'One of the following: quasi_IPL, '
                                      'random_noise, noisy'))
    parser.add_option('-t', '--method2', dest='method2', default='Mproper',
                      type=str, help=('Method to impute the matrix M.'
                                      'One of the following: Mproper'))
    return parser.parse_args()


###############################################################################
# ## MAIN #####################################################################
###############################################################################
def main(problems, ns, nf, n_classes, n_sim, loss, rho, n_it, method, method2):

    problem_list = problems.split(',')

    for problem in problem_list:
        np.random.seed(42)
        ############################
        # ## Create a Diary for all the logs and results
        diary = Diary(name='testWLCkeras', path='results', overwrite=False,
                      image_format='png', fig_format='svg')
        diary.add_notebook('dataset')
        diary.add_notebook('validation')

        ############################
        # ## Configurable parameters
        # Parameters of the weak label model
        alpha = 0.8
        beta = 0.2
        gamma = 0.2

        n_epoch = 20
        #####################
        # ## A title to start

        print "======================================"
        print "    Testing Learning from Weak Labels "
        print "======================================"

        #######################################################################
        # ## PART I: Load data (samples and true labels)                     ##
        #######################################################################

        # X, y = skd.make_classification(
        #     n_samples=ns, n_features=nf, n_informative=3, n_redundant=0,
        #     n_repeated=0, n_classes=n_classes, n_clusters_per_class=2,
        #     weights=None, flip_y=0.0001, class_sep=1.0, hypercube=True,
        #     shift=0.0, scale=1.0, shuffle=True, random_state=None)
        if problem in openml_ids.keys():
            dataset_id = openml_ids[problem]
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, categorical = dataset.get_data(
                                    target=dataset.default_target_attribute,
                                    return_categorical_indicator=True)
            # TODO change NaN in categories for another category
            enc = OneHotEncoder(categorical_features=categorical, sparse=False)
            X = enc.fit_transform(X)  # Categorical to binary
            ns = X.shape[0]           # Sample size
            nf = X.shape[1]             # Data dimension
            n_classes = y.max()+1      # Number of classes
            n_it = 20            # Number of iterations
        elif problem == 'blobs':
            X, y = skd.make_blobs(n_samples=ns, n_features=nf,
                                  centers=n_classes, cluster_std=1.0,
                                  center_box=(-10.0, 10.0), shuffle=True,
                                  random_state=None)
        elif problem == 'gauss_quantiles':
            X, y = skd.make_gaussian_quantiles(n_samples=ns, n_features=nf,
                                               n_classes=n_classes,
                                               shuffle=True, random_state=None)
        elif problem == 'digits':
            X, y = skd.load_digits(n_class=n_classes, return_X_y=True)
            nf = X.shape[0]             # Data dimension
            n_it = 10            # Number of iterations
        else:
            raise("Problem type unknown: {}".format(problem))
        X = Imputer(missing_values='NaN', strategy='mean').fit_transform(X)
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

        # Convert y into a binary matrix
        y_bin = label_binarize(y, range(n_classes))

        # ## Report data used in the simulation
        print '----------------'
        print 'Simulation data:'
        print '    Dataset name: {0}'.format(problem)
        print '    Sample size: {0}'.format(ns)
        print '    Number of features: {0}'.format(nf)
        print '    Number of classes: {0}'.format(n_classes)

        diary.add_entry('dataset', ['name', problem, 'size', ns,
                                    'n_features', nf, 'n_classes', n_classes,
                                    'method', method, 'method2', method2])

        # Generate weak labels
        M = wlw.computeM(n_classes, alpha=alpha, beta=beta, gamma=gamma,
                         method=method)
        z = wlw.generateWeak(y, M, n_classes)
        v = wlw.computeVirtual(z, n_classes, method=method)
        v2 = wlw.computeVirtual(z, n_classes, method=method2, M=M)

        # Convert z to a list of binary lists (this is for the OSL alg)
        z_bin = wlw.computeVirtual(z, n_classes, method='IPL')

        # If dimension is 2, we draw a scatterplot
        if nf >= 2:
            fig = plot_data(X, y, save=False)
            diary.save_figure(fig)

        ######################
        # ## Select classifier

        ######################################################################
        # ## PART II: AL algorithm analysis                                ###
        ######################################################################

        print '----------------------------'
        print 'Weak Label Analysis'

        wLR = {}
        title = {}
        n_jobs = {}
        v_dict = {}
        Pe_tr = {}
        Pe_cv = {}
        Pe_tr_mean = {}
        Pe_cv_mean = {}
        params = {'rho': rho, 'n_it': n_it, 'loss': loss}
        tag_list = []

        # ###################
        # Supervised learning
        tag = 'Supervised'
        title[tag] = 'Learning from clean labels:'
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL',
                                              optimizer='GD', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = y
        tag_list.append(tag)

        # ##########################
        # Supervised learning (BFGS)
        tag = 'Superv-BFGS'
        title[tag] = 'Learning from clean labels with BFGS:'
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL',
                                              optimizer='BFGS', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = y
        tag_list.append(tag)

        # ##################################
        # Optimistic Superset Learning (OSL)
        tag = 'OSL'
        title[tag] = 'Optimistic Superset Loss (OSL)'
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL',
                                              optimizer='GD', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = z_bin
        tag_list.append(tag)

        # ############################################
        # Optimistic Superset Learning (OSL) with BFGS
        tag = 'OSL-BFGS'
        title[tag] = 'Optimistic Superset Loss (OSL) with BFGS'
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='OSL',
                                              optimizer='BFGS', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = z_bin
        tag_list.append(tag)

        # # ############################################
        # # Add hoc M-proper loss with Gradient Descent
        tag = 'Mproper-GD'
        title[tag] = 'M-proper loss with Gradient Descent'
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                              optimizer='GD', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = v2
        tag_list.append(tag)

        # # ############################################
        # # Add hoc M-proper loss with BFGS
        tag = 'Mproper-BFGS'
        title[tag] = 'M-proper loss with Gradient Descent'
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                              optimizer='BFGS', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = v2
        tag_list.append(tag)

        # ############################################
        # Virtual Label Learning with Gradient Descent
        tag = 'VLL-GD'
        title[tag] = 'Virtual Label Learning (VLL) with Gradient Descent'
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                              optimizer='GD', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = v
        tag_list.append(tag)

        # ###################################################
        # Virtual Label Learning with BFGS and regularization
        tag = 'VLL-BFGS'
        title[tag] = 'Virtual Label Learning (VLL) with BFGS and regularization'
        params = {'alpha': (2.0 + nf)/2, 'loss': loss}    # This alpha is an heuristic
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                              optimizer='BFGS', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = v
        tag_list.append(tag)

        # ############################################
        # Virtual Label Learning with Gradient Descent
        tag = 'VLLc-GD'
        title[tag] = 'CC-VLL with Gradient Descent'
        params = {'rho': rho, 'n_it': n_it, 'loss': loss}
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                              optimizer='GD', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = z_bin
        tag_list.append(tag)

        # ############################################
        # Virtual Label Learning with Gradient Descent
        tag = 'VLLc-BFGS'
        title[tag] = 'CC-VLL with BFGS'
        wLR[tag] = wlc.WeakLogisticRegression(n_classes, method='VLL',
                                              optimizer='BFGS', params=params)
        n_jobs[tag] = -1
        v_dict[tag] = z_bin
        tag_list.append(tag)

        # ############################################
        # Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
        tag = 'Keras-LR-Superv-SGD'
        title[tag] = 'Keras M-proper loss with Stochastic Gradient Descent'
        params = {'n_epoch': n_epoch}
        wLR[tag] = km.KerasWeakLogisticRegression(input_size=X.shape[1],
                                                  output_size=n_classes,
                                                  optimizer='SGD',
                                                  params=params)
        n_jobs[tag] = -1
        v_dict[tag] = y_bin
        tag_list.append(tag)

        # ############################################
        # Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
        tag = 'Keras-LR-OSL-SGD'
        title[tag] = 'Keras OSL loss with Stochastic Gradient Descent'
        params = {'n_epoch': n_epoch}
        wLR[tag] = km.KerasWeakLogisticRegression(input_size=X.shape[1],
                                                  output_size=n_classes,
                                                  optimizer='SGD',
                                                  OSL=True,
                                                  params=params)
        n_jobs[tag] = -1
        v_dict[tag] = z_bin
        tag_list.append(tag)

        # ############################################
        # Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
        tag = 'Keras-LR-QIPL-SGD'
        title[tag] = 'Keras Logistic regression QIPL loss with Stochastic GD'
        params = {'n_epoch': n_epoch}
        wLR[tag] = km.KerasWeakLogisticRegression(input_size=X.shape[1],
                                                  output_size=n_classes,
                                                  optimizer='SGD',
                                                  params=params)
        n_jobs[tag] = -1
        v_dict[tag] = v
        tag_list.append(tag)

        # ############################################
        # Miquel: Add hoc M-proper loss with Stochastic Gradient Descent
        tag = 'Keras-LR-Mproper-SGD'
        title[tag] = 'Keras Logistic regression M-proper loss with Stochastic GD'
        params = {'n_epoch': n_epoch}
        wLR[tag] = km.KerasWeakLogisticRegression(input_size=X.shape[1],
                                                  output_size=n_classes,
                                                  optimizer='SGD',
                                                  params=params)
        n_jobs[tag] = -1
        v_dict[tag] = v2
        tag_list.append(tag)

        # ############################################
        # Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
        tag = 'Keras-MLP-Superv-SGD'
        title[tag] = 'Keras MLP OSL loss with Stochastic Gradient Descent'
        params = {'n_epoch': n_epoch}
        wLR[tag] = km.KerasWeakMultilayerPerceptron(input_size=X.shape[1],
                                                    output_size=n_classes,
                                                    optimizer='SGD',
                                                    OSL=True,
                                                    params=params)
        n_jobs[tag] = 1
        v_dict[tag] = y_bin
        tag_list.append(tag)

        # ############################################
        # Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
        tag = 'Keras-MLP-OSL-SGD'
        title[tag] = 'Keras MLP OSL loss with Stochastic Gradient Descent'
        params = {'n_epoch': n_epoch}
        wLR[tag] = km.KerasWeakMultilayerPerceptron(input_size=X.shape[1],
                                                    output_size=n_classes,
                                                    optimizer='SGD',
                                                    OSL=True,
                                                    params=params)
        n_jobs[tag] = 1
        v_dict[tag] = z_bin
        tag_list.append(tag)

        # ############################################
        # Miquel: Add hoc Supervised loss with Stochastic Gradient Descent
        tag = 'Keras-MLP-QIPL-SGD'
        title[tag] = 'Keras MLP QIPL loss with Stochastic Gradient Descent'
        wLR[tag] = km.KerasWeakMultilayerPerceptron(input_size=X.shape[1],
                                                    output_size=n_classes,
                                                    optimizer='SGD',
                                                    params=params)
        n_jobs[tag] = 1
        v_dict[tag] = v
        tag_list.append(tag)

        # ############################################
        # Miquel: Add hoc M-proper loss with Stochastic Gradient Descent
        tag = 'Keras-MLP-Mproper-SGD'
        title[tag] = 'Keras MLP M-proper loss with Stochastic Gradient Descent'
        params = {'n_epoch': n_epoch}
        wLR[tag] = km.KerasWeakMultilayerPerceptron(input_size=X.shape[1],
                                                    output_size=n_classes,
                                                    optimizer='SGD',
                                                    params=params)
        n_jobs[tag] = 1
        v_dict[tag] = v2
        tag_list.append(tag)

        # ############
        # Evaluation and plot of each model
        appended_dfs = []
        for i, tag in enumerate(tag_list):
            print tag
            Pe_tr[tag], Pe_cv[tag] = evaluateClassif(wLR[tag], X, y,
                                                     v_dict[tag], n_sim=n_sim,
                                                     n_jobs=n_jobs[tag])
            fig = plot_results(tag_list[:(i+1)], Pe_tr, Pe_cv, ns, n_classes,
                               n_sim, save=False)
            diary.save_figure(fig)

            rows = [[tag, title[tag], n_jobs[tag], loss, j, tr_l, cv_l]
                    for j, (tr_l, cv_l) in enumerate(zip(Pe_tr[tag], Pe_cv[tag]))]
            df_aux = pd.DataFrame(rows, columns=['tag', 'title', 'jobs',
                                                 'loss', 'sim', 'loss_train',
                                                 'loss_val'])
            appended_dfs.append(df_aux)

        df = pd.concat(appended_dfs, axis=0, ignore_index=True)
        df.to_csv(os.path.join(diary.path, 'pd_df_results.csv'))

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

if __name__ == '__main__':
    (options, args) = parse_arguments()

    main(**vars(options))
