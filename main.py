#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code evaluates logistic regression with weak labels

    Author: JCS, June, 2016
            MPN, April, 2017 (Updated October, 2020)
"""

# External modules
import os
import warnings
import time

import pandas as pd
import numpy as np
import openml
from optparse import OptionParser
from sklearn.preprocessing import label_binarize
# import sklearn.linear_model as sklm

# My modules
import weaklabels.WLclassifier as wlc
import weaklabels.WLweakener as wlw

import utils.keras_models as km
from utils.data import load_dataset
from utils.evaluation import evaluateClassif
from utils.visualization import plot_data, plot_results

from utils.diary import Diary

warnings.filterwarnings("ignore")
seed = 42


def parse_arguments():
    parser = OptionParser()
    parser.add_option('-p', '--datasets', dest='datasets', default='blobs',
                      type=str, help=('List of datasets or toy examples to'
                                      'test separated by with no spaces.'))
    # Parameters for sklearn synthetic data
    parser.add_option('-s', '--n-samples', dest='ns', default=1000,
                      type=int, help='Number of samples if toy dataset.')
    parser.add_option('-f', '--n-features', dest='nf', default=2,
                      type=int, help='Number of features if toy dataset.')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=5,
                      type=int, help='Number of classes if toy dataset.')
    # Common parameters for all AL algorithms
    parser.add_option('-m', '--n-simulations', dest='n_sim', default=10,
                      type=int, help='Number of times to run every model.')
    parser.add_option('-l', '--loss', dest='loss', default='CE',
                      type=str, help=('Loss function to minimize between '
                                      'square (brier score) or CE (cross '
                                      'entropy)'))
    parser.add_option('-u', '--path-results', dest='path_results',
                      default='results', type=str,
                      help=('Path to save the results'))
    # Parameters of the classiffier fit method
    parser.add_option('-r', '--rho', dest='rho', default=0.0002,
                      type=float,
                      help='Learning step for the Gradient Descent')
    parser.add_option('-a', '--alpha', dest='alpha', default=0.5,
                      type=float,
                      help='Alpha probability parameter')
    parser.add_option('-b', '--beta', dest='beta', default=0.5,
                      type=float,
                      help='Beta probability parameter')
    parser.add_option('-i', '--n-iterations', dest='n_it', default=10,
                      type=int, help=('Number of iterations of '
                                      'Gradient Descent.'))
    parser.add_option('-e', '--mixing-matrix', default='quasi-IPL',
                      type=str, help=('Method to generate the mixing matrix M.'
                                      'One of the following: IPL, quasi-IPL, '
                                      'noisy, random_noise, random_weak'))
    parser.add_option('-w', '--classifier_name', default='LR',
                      type=str, help=('Classification method between '
                                      'LR (logistic regression) or FNN (feedforward '
                                      'neural network)'))
    parser.add_option('-o', '--optimizer', default='SGD',
                      type=str, help=('Gradient descent method between '
                                      'SGD (stochastic gradient descent) or BFGS '))
    return parser.parse_args()

def run_experiment(dataset, ns, nf, n_classes, n_sim, loss, rho, n_it,
                   mixing_matrix,
                   alpha, beta, path_results, classifier_name, optimizer):
    np.random.seed(seed)
    ############################
    # ## Create a Diary for all the logs and results
    diary = Diary(name='main', path=path_results, overwrite=False,
                  image_format='png', fig_format='svg')
    diary.add_notebook('dataset')
    diary.add_notebook('validation')

    #####################
    # ## A title to start

    print("======================================")
    print("    Testing Learning from Weak Labels ")
    print("======================================")

    #######################################################################
    # ## PART I: Load data (samples and true labels)                     ##
    #######################################################################
    X, y, n_classes, n_samples, n_features = load_dataset(dataset,
                                                          n_samples=ns,
                                                          n_features=nf,
                                                          n_classes=n_classes,
                                                          seed=seed)
    diary.add_entry('dataset', ['name', dataset, 'size', n_samples,
                                'n_features', n_features, 'n_classes',
                                n_classes, 'mixing_matrix', mixing_matrix,
                                'alpha', alpha, 'beta', beta])

    # Convert y into a binary matrix
    y_bin = label_binarize(y, classes = list(range(n_classes)))

    # Generate weak labels
    WLM = wlw.WLmodel(n_classes, model_class=mixing_matrix)
    M = WLM.generateM(alpha=alpha, beta=beta)
    WLM.remove_zero_rows()
    z = WLM.generateWeak(y)
    # Compute the virtual labels
    v_methods = ['binary','quasi-IPL','M-pinv','M-conv','M-opt','M-opt-conv']
    
    v = {}
    for v_method in v_methods:
        v[v_method] = WLM.virtual_labels(z, method=v_method)

    # Convert z to a list of binary lists (this is for the OSL alg)
    z_bin = wlw.binarizeWeakLabels(z, n_classes)
    z_bin_oh = label_binarize(z, classes=WLM.weak_classes)
    # TODO Add to diary: p = np.sum(weak_labels,0)/np.sum(weak_labels)

    #######################################
    # Explanation of the different labels
    # e.g. n_classes = 3
    #   y: (true labels) array of integers with shape (n_samples, )
    #       e.g. [2, 0, 1]
    #   y_bin: (true labels) matrix of integers with shape (n_samples,
    #       n_classes) with the one hot encoding of y
    #       e.g.[[0, 0, 1],
    #            [1, 0, 0],
    #            [0, 1, 0]]
    #   z: (weak labels) array of integers with shape (n_samples, )
    #       generated randomly using the mixing matrix M
    #       e.g. [1, 4, 6]
    #   FIXME z_bin are floats
    #   z_bin: (weak labels) matrix of integers with shape (n_samples,
    #       n_classes) with the binary encoding of z
    #       eg[[0, 0, 1],
    #          [1, 0, 0],
    #          [1, 1, 0]]
    #   v: (virtual labels) matrix of floats with shape (n_samples,
    #       n_classes) with the binary encoding of v
    #       e.g.[[0., 0., 1.],
    #            [1., 0., 0.],
    #            [1., 1., -1.]]
    #   v2: (virtual labels) matrix of floats with shape (n_samples,
    #       n_classes) using the M-proper and the known mixing matrix M
    #       e.g. [[-0.07, -0.07, 1.34],
    #             [1.34, -0.07, -0.07],
    #             [0.31, 0.31, -0.03]]
    ### e.g.
    ## n_classes = 3
    ## y = np.array([2,0,1])
    ## y_bin = label_binarize(y, range(n_classes))
    ## M = wlw.computeM(n_classes, alpha=alpha, beta=beta, method=mixing_matrix)
    ## z = wlw.generateWeak(y, M, n_classes)
    ## # TODO if this is not to compute virtual it shouldn't be called the
    ## #  same function. It should be dec to bin or seomething like that
    ## z_bin = wlw.dec_to_bin(z, n_classes)
    ## v = wlw.computeVirtual(z, n_classes, method=mixing_matrix)

    # If dimension is >=2, we draw a scatterplot
    if nf >= 2:
        fig = plot_data(X, y, save=False, title=dataset)
        diary.save_figure(fig, filename=dataset)

        if M.shape[0] == M.shape[1]:
            fig = plot_data(X, n_classes-np.log(z)-1, save=False, title=dataset)
            diary.save_figure(fig, filename='{}_{}'.format(dataset,
                                                           mixing_matrix))


    ######################
    # ## Select classifier

    ######################################################################
    # ## PART II: AL algorithm analysis                                ###
    ######################################################################
    print('----------------------------')
    print('Weak Label Analysis')

    clf_dict = {}
    title = {}
    n_jobs = {}
    v_dict = {}
    Pe_tr = {}  # training performance
    Pe_cv = {}  # cross-validation performance
    Pe_tr_mean = {}
    Pe_cv_mean = {}
    params = {'rho': rho, 'n_it': n_it, 'loss': loss}
    params_keras = {'n_epoch': n_it, 'random_seed': 0}
    tag_list = []
    classifier_dict = {'LR': km.KerasWeakLogisticRegression,
                       'FNN': km.KerasWeakMultilayerPerceptron}
    #######################################################################
    # BASELINES:
    # 1. Upper bound: Training with Original labels
    #
    # ###################
    # Supervised learning

    #optimizer = 'SGD' # SGD or BFGS
    #classifier_name = 'LR' # LR or FNN
    classifier =  classifier_dict[classifier_name]
    # ############################################
    # Supervised loss with Stochastic Gradient Descent
    tag = '{}_Superv_{}'.format(classifier_name, optimizer)
    title[tag] = '{} trained with true labels with {}'.format(classifier_name,
                                                              optimizer)
    clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
                          optimizer=optimizer, params=params_keras, loss_f = loss)
    n_jobs[tag] = 1
    v_dict[tag] = y_bin.astype(float)
    tag_list.append(tag)

    # ############################################
    # 2. Lower bound: Training with weak labels
    # 2.a : Also case where we assume IPL mixing matrix M
    #
    # ############################################
    # Training with the weak labels with Stochastic Gradient Descent
    tag = '{}_Weak_{}'.format(classifier_name, optimizer)
    title[tag] = '{} trained with weak labels with {}'.format(classifier_name,
                                                              optimizer)
    clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
                          optimizer=optimizer, params=params_keras, loss_f = loss)
    n_jobs[tag] = 1
    v_dict[tag] = z_bin
    tag_list.append(tag)

    # ############################################
    # 3. Competitor: Optimistic Superset Learning and weak labels
    #
    # ##################################
    # Optimistic Superset Learning (OSL)
    ## FIXME There is a problem with argmax() and argument keep_dims
    #tag = '{}-OSL-{}'.format(classifier_name, optimizer)
    #title[tag] = '{} OSL loss with {}'.format(classifier_name, optimizer)
    #clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
    #                      optimizer=optimizer, OSL=True, params=params_keras)
    #n_jobs[tag] = 1
    #v_dict[tag] = z_bin
    #tag_list.append(tag)

    # ############################################
    # 4. Competitor: EM algorithm
    #
    # ##################################
    # Eexpectation maximization algorithm (EM)
    tag = '{}_EM_{}'.format(classifier_name, optimizer)
    title[tag] = '{} OSL loss with {}'.format(classifier_name, optimizer)
    clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
                          optimizer=optimizer, params=params_keras, loss_f=loss, EM=True)
    n_jobs[tag] = 1
    v_dict[tag] = z_bin_oh @ M
    tag_list.append(tag)

    #######################################################################
    # OUR PROPOSED METHODS:
    # 1. Upper bound (if we know M): Training with Mproper virtual labels
    #
    # ############################################
    # Known M with pseudoinverse
    v_method = 'M-pinv'
    tag = '{}_{}_{}'.format(classifier_name, v_method, optimizer)
    title[tag] = '{} {} with {}'.format(classifier_name, v_method, optimizer)
    clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
                          optimizer=optimizer, params=params_keras, loss_f = loss)
    n_jobs[tag] = 1
    v_dict[tag] = v[v_method]
    tag_list.append(tag)

    # ############################################
    # Known M with convexity restriction
    v_method = 'M-conv'
    tag = '{}_{}_{}'.format(classifier_name, v_method, optimizer)
    title[tag] = '{} {} with {}'.format(classifier_name,
                                                       v_method,
                                                       optimizer)
    clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
                          optimizer=optimizer, params=params_keras, loss_f = loss)
    n_jobs[tag] = 1
    v_dict[tag] = v[v_method]
    tag_list.append(tag)

    # ############################################
    # Known M with new method convex
    v_method = 'M-opt'
    tag = '{}_{}_{}'.format(classifier_name, v_method, optimizer)
    title[tag] = '{} {} and new method with {}'.format(classifier_name,
                                                       v_method,
                                                       optimizer)
    clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
                          optimizer=optimizer, params=params_keras, loss_f = loss)
    n_jobs[tag] = 1
    v_dict[tag] = v[v_method]
    tag_list.append(tag)

    # ############################################
    # Known M with new method convex
    v_method = 'M-opt-conv'
    tag = '{}_{}_{}'.format(classifier_name, v_method, optimizer)
    title[tag] = '{} {} and new method with {}'.format(classifier_name,
                                                       v_method,
                                                       optimizer)
    clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
                          optimizer=optimizer, params=params_keras, loss_f = loss)
    n_jobs[tag] = 1
    v_dict[tag] = v[v_method]
    tag_list.append(tag)
    
    # ############################################
    # 2. If we assume quasi-IPL M: Training with virtual labels for q_IPL
    #
    # ############################################
    # Virtual labels assuming quasi-IPL loss with SGD
    #v_method = 'quasi-IPL'
    #tag = '{}-{}-{}'.format(classifier_name, v_method, optimizer)
    #title[tag] = '{} {} with {}'.format(classifier_name, v_method, optimizer)
    #clf_dict[tag] = classifier(input_size=X.shape[1], output_size=n_classes,
    #                      optimizer=optimizer, params=params_keras)
    #n_jobs[tag] = 1
    #v_dict[tag] = v[v_method]
    #tag_list.append(tag)

    # ############
    # Evaluation and plot of each model
    appended_dfs = []
    for i, tag in enumerate(tag_list):
        print(tag)
        t_start = time.time()
        Pe_tr[tag], Pe_cv[tag] = evaluateClassif(clf_dict[tag], X, y,
                                                 v_dict[tag], n_sim=n_sim,
                                                 n_jobs=n_jobs[tag])
        seconds = time.time() - t_start
        fig = plot_results(tag_list[:(i+1)], Pe_tr, Pe_cv, ns, n_classes,
                           n_sim, save=False)
        diary.save_figure(fig)

        rows = [[seconds, tag, title[tag], n_jobs[tag], loss,
                 j, tr_l, cv_l]
                    for j, (tr_l, cv_l) in enumerate(zip(Pe_tr[tag], Pe_cv[tag]))]
        df_aux = pd.DataFrame(rows, columns=['seconds', 'tag', 'title',
                                             'jobs', 'loss', 'sim',
                                             'loss_train', 'loss_val'])
        appended_dfs.append(df_aux)

    df = pd.concat(appended_dfs, axis=0, ignore_index=True)
    df.to_csv(os.path.join(diary.path, 'pd_df_results.csv'))

    # ############
    # Print results.
    for tag in tag_list:
        Pe_tr_mean[tag] = np.mean(Pe_tr[tag])
        Pe_cv_mean[tag] = np.mean(Pe_cv[tag])

        print(title[tag])
        print('* Average train error = {0}'.format(Pe_tr_mean[tag]))
        print('* Average cv error = {0}'.format(Pe_cv_mean[tag]))

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

    print('================')
    print('Fin de ejecucion')

###############################################################################
# ## MAIN #####################################################################
###############################################################################
def main(datasets, ns, nf, n_classes, n_sim, loss, rho, n_it, mixing_matrix,
         alpha, beta, path_results,classifier_name, optimizer):

    dataset_list = datasets.split(',')
    mixing_matrix_list = mixing_matrix.split(',')

    for dataset in dataset_list:
        for mixing_matrix in mixing_matrix_list:
            run_experiment(dataset, ns, nf, n_classes, n_sim, loss, rho, n_it,
                           mixing_matrix, alpha, beta, path_results,classifier_name, optimizer)

if __name__ == '__main__':
    (options, args) = parse_arguments()

    main(**vars(options))
