#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code contains a method to load datasets for testing classification
    algorithns 

    Author: JCS, June, 2016
"""

# External modules
import sklearn.datasets as skd


def getDataset(name):

    # This method provides the feature matrix (X) and the target variables (y)
    # of different datasets for multiclass classification
    # Args:
    #    name: A tag name for a specific dataset
    #
    # Returns:
    #    X:    Input data matrix
    #    y:    Target array.

    
    if name == 'hypercube':
        X, y = skd.make_classification(
            n_samples=400, n_features=40, n_informative=40, 
            n_redundant=0, n_repeated=0, n_classes=4, 
            n_clusters_per_class=2,
            weights=None, flip_y=0.0001, class_sep=1.0, hypercube=True,
            shift=0.0, scale=1.0, shuffle=True, random_state=None)

    elif name == 'blobs':
        X, y = skd.make_blobs(
            n_samples=400, n_features=2, centers=20, cluster_std=2,
            center_box=(-10.0, 10.0), shuffle=True, random_state=None)

    elif name == 'blobs2':
        X, y = skd.make_blobs(
            n_samples=400, n_features=4, centers=10, cluster_std=1,
            center_box=(-10.0, 10.0), shuffle=True, random_state=None)

    elif name == 'iris':
        dataset = skd.load_iris()
        X = dataset.data
        y = dataset.target

    elif name == 'digits':
        dataset = skd.load_digits()
        X = dataset.data
        y = dataset.target

    elif name == 'covtype':
        dataset = skd.fetch_covtype()
        X = dataset.data
        y = dataset.target - 1

    return X, y


