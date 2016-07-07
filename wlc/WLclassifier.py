#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Defines classifier objects that work with weak labels

    Author: JCS, May. 2016
"""

import numpy as np
import ipdb


class WeakLogisticRegression(object):

    def __init__(self, n_classes=2, rho=0.1, n_it=100, sound='off'):

        """
        Only a name is needed when the object is created
        """

        self.sound = sound
        self.rho = rho
        self.n_it = n_it
        self.n_classes = n_classes
        self.classes_ = range(n_classes)

    def softmax(self, x):
        """
        Computes the softmax transformation

        Args:
            :x  : NxC matrix of N samples with dimension C

        Returns:
            :p  : NxC matrix of N probability vectors with dimension C
        """

        p = np.exp(x)
        p = p / np.sum(p, axis=1, keepdims=True)

        return p

    def index2bin(self, vector, dim):
        """ Converts an array of indices into a matrix of binary vectors

            Adapted from "http://stackoverflow.com/questions/23300715/
                          numpy-transform-vector-to-binary-matrix"
            (Check the web link to see a faster sparse version that is much
            more efficient for large dimensions)

            Args:
               :vector: Array of integer indices 0, 1, ..., dim-1
               :dim: Dimension of the output vector.
        """

        n = vector.shape[0]
        v_bin = np.zeros((n, dim))
        v_bin[np.arange(n), vector] = 1

        return v_bin

    def fit(self, X, Y):
        """
        Fits a logistic regression model to instances in X given
        the labels in Y

        Args:
            :X :Input data, numpy array of shape[n_samples, n_features]
            :Y :Target for X, with shape [n_samples].
                Each target can be a index in [0,..., self.n_classes-1] or
                a binary vector with dimension self.n_classes

        Returns:
            :self
        """
   
        # Data dimension
        n_dim = X.shape[1]

        # If labels are 1D, transform them into binary label vectors
        if len(Y.shape) == 1:

            # If the alphabet is not [0, 1, ..., n_classes-1] transform
            # labels into these values.
            # if not(set(self.classes_) < set(xrange(self.n_classes))):
            #     alphabet_inv = dict(zip(self.classes_, range(self.n_classes)))
            #     Y0 = np.array([alphabet_inv[c] for c in Y])
            # else:
            #     Y0 = Y

            T = self.index2bin(Y, self.n_classes)

        else:
            T = Y

        # Initialize variables
        # pe_tr = np.zeros(self.n_it)
        W = np.random.randn(n_dim, self.n_classes)

        # Running the gradient descent algorithm
        for n in range(self.n_it):

            # Compute posterior probabilities for weight w
            p = self.softmax(np.dot(X, W))

            # Update weights
            W += self.rho*np.dot(X.T, T - p)

        self.W = W

        return self    # w, nll_tr

    def predict(self, X):

        # Compute posterior probability of class 1 for weights w.
        p = self.softmax(np.dot(X, self.W))

        # Class
        D = np.argmax(p, axis=1)

        return D  # p, D

    def predict_proba(self, X):

        # Compute posterior probability of class 1 for weights w.
        p = (np.c_[self.softmax(np.dot(X, self.W))])
        return p

    def get_params(self, deep=True):

        # suppose this estimator has parameters "alpha" and "recursive"
        return {"n_classes": self.n_classes, "sound": self.sound, "rho": self.rho, "n_it": self.n_it}
