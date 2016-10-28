#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Defines classifier objects that work with weak labels

    Author: JCS, May. 2016
"""

import numpy as np
import scipy as sp
import ipdb


class WeakLogisticRegression(object):

    def __init__(self, n_classes=2, method="VLL", optimizer='GD',
                 params={}, sound='off'):

        """
        Only a name is needed when the object is created
        """

        self.sound = sound
        self.params = params
        self.method = method
        self.optimizer = optimizer
        self.n_classes = n_classes
        self.classes_ = range(n_classes)

        # Set default value of parameter 'alpha' if it does not exist
        if self.method == "VLL" and 'alpha' not in self.params:
            self.params['alpha'] = 0
        if self.method == "Mproper" and 'alpha' not in self.params:
            self.params['alpha'] = 0

    def softmax(self, x):
        """
        Computes the softmax transformation

        Args:
            :x  : NxC matrix of N samples with dimension C

        Returns:
            :p  : NxC matrix of N probability vectors with dimension C
        """

        # Shift each x to have maximum value=1. This is to avoid exponentiating
        # large numbers that cause overflow
        z = x - np.max(x, axis=1, keepdims=True)
        p = np.exp(z)
        p = p / np.sum(p, axis=1, keepdims=True)

        return p

    def logsoftmax(self, x):
        """
        Computes the elementwise logarithm of the softmax transformation

        Args:
            :x  : NxC matrix of N samples with dimension C

        Returns:
            :p  : NxC matrix of N probability vectors with dimension C
        """

        # Shift each x to have maximum value=1. This is to avoid exponentiating
        # large numbers that cause overflow
        z = x - np.max(x, axis=1, keepdims=True)
        logp = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))

        return logp

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

    def hardmax(self, Z):

        """ Transform each row in array Z into another row with zeroes in the
            non-maximum values and 1/nmax on the maximum values, where nmax is
            the number of elements taking the maximum value
        """

        D = sp.equal(Z, np.max(Z, axis=1, keepdims=True))

        # In case more than one value is equal to the maximum, the output
        # of hardmax is nonzero for all of them, but normalized
        D = D/np.sum(D, axis=1, keepdims=True)

        return D

    def logLoss(self, w, X, T):

        """ Compute a regularized log loss (cross-entropy) for samples in X
            virtual labels in T and parameters w. The regularization parameter
            is taken from the object attributes.
            It assumes a multi-class softmax classifier.
            This method implements two different log-losses, that are specified
            in the object's attribute self.method:
                'OSL' :Optimistic Superset Loss. It assumes that the true label
                       is the nonzero weak label with highest posterior
                       probability given the model.
                'VLL' :Virtual Labels Loss.
            The regularization parameter is set in set.params['alpha']

            Args:
                :w:  1-D nympy array. It is a flattened versio of the weigh
                     matrix of the multiclass softmax. This 1-D arrangement
                     is required by the scipy optimizer that will use this
                     method.
                :X:  Input data. An (NxD) matrix of N samples with dimension D
                :T:  Target class. An (NxC) array of target vectors.
                     The meaning and format of each target vector t depends
                     on the selected log-loss version:
                     - 'OSL': t is a binary vector.
                     - 'VLL': t is a virtual label vector
            Returns:
                :L:  Log-loss
        """

        n_dim = X.shape[1]
        W2 = w.reshape((n_dim, self.n_classes))
        logp = self.logsoftmax(np.dot(X, W2))

        if self.method == 'OSL':
            p = np.exp(logp)
            D = self.hardmax(T*p)
            L = -np.sum(D*logp)

        else:
            # Bias: This term is usually zero for proper losses, but may be
            # nonzero for RC or CC weak losses
            # Note, also, that the bias could be computed out of this function.
            bias = np.sum(1 - np.sum(T, axis=1))
            L = -np.sum(T*logp) + bias + self.params['alpha']*np.sum(w**2)/2
            # L = -np.sum(T*logp)

        if L < 0:
            print "Oooops, negative log loss. ",
            print "You should user a larger parameter alpha"
            print "L = {0}".format(L)
            ipdb.set_trace()

        return L

    def gradLogLoss(self, w, X, T):

        """ Compute the gradient of the regularized log loss (cross-entropy)
            for samples in X, virtual labels in T and parameters w.
            The regularization parameter is taken from the object attributes.
            It assumes a multi-class softmax classifier.
            This method implements gradients for two different log-losses, that
            are specified in the object's attribute self.method:
                'OSL' :Optimistic Superset Loss. It assumes that the true label
                       is the nonzero weak label with highest posterior
                       probability given the model.
                'VLL' :Virtual Labels Loss.
            The regularization parameter is set in set.params['alpha']

            Args:
                :w:  1-D nympy array. It is a flattened versio of the weigh
                     matrix of the multiclass softmax. This 1-D arrangement
                     is required by the scipy optimizer that will use this
                     method.
                :X:  Input data. An (NxD) matrix of N samples with dimension D
                :T:  Target class. An (NxC) array of target vectors.
                     The meaning and format of each target vector t depends
                     on the selected log-loss version:
                     - 'OSL': t is a binary vector.
                     - 'VLL': t is a virtual label vector
            Returns:
                :G:  Gradient of the Log-loss
        """

        n_dim = X.shape[1]
        W2 = w.reshape((n_dim, self.n_classes))
        p = self.softmax(np.dot(X, W2))

        if self.method == 'OSL':
            D = self.hardmax(T*p)
            G = np.dot(X.T, p - D)

        else:
            bias = np.sum(T, axis=1, keepdims=True)
            G = np.dot(X.T, p*bias - T) - self.params['alpha']*W2
            # G = np.dot(X.T, p - T)

        return G.reshape((n_dim*self.n_classes))

    def gd(self, X, T):

        """ Trains a logistic regression classifier by a gradient descent
            method
        """

        # Initialize variables
        n_dim = X.shape[1]
        W = np.random.randn(n_dim, self.n_classes)
        w1 = W.reshape((n_dim*self.n_classes))

        # Running the gradient descent algorithm
        for n in range(self.params['n_it']):

            w1 = W.reshape((n_dim*self.n_classes))

            G = self.gradLogLoss(w1, X, T).reshape((n_dim, self.n_classes))

            W -= self.params['rho']*G

        return W

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

        self.n_dim = X.shape[1]

        # If labels are 1D, transform them into binary label vectors
        if len(Y.shape) == 1:

            # If the alphabet is not [0, 1, ..., n_classes-1] transform
            # labels into these values.
            # if not(set(self.classes_) < set(xrange(self.n_classes))):
            #     alphabet_inv = dict(zip(self.classes_,range(self.n_classes)))
            #     Y0 = np.array([alphabet_inv[c] for c in Y])
            # else:
            #     Y0 = Y

            T = self.index2bin(Y, self.n_classes)

        else:
            T = Y

        # Optimization
        if self.optimizer == 'GD':
            self.W = self.gd(X, T)
        else:
            w0 = 1*np.random.randn(X.shape[1]*self.n_classes)
            res = sp.optimize.minimize(
                self.logLoss, w0, args=(X, T), method=self.optimizer,
                jac=self.gradLogLoss, hess=None, hessp=None, bounds=None,
                constraints=(), tol=None, callback=None,
                options={'disp': False, 'gtol': 1e-20,
                         'eps': 1.4901161193847656e-08, 'return_all': False,
                         'maxiter': None, 'norm': np.inf})
            #    options=None)
            self.W = res.x.reshape((self.n_dim, self.n_classes))

            if res.status != 0:
                print "{0}-{1}: Status {2}. {3}. {4}".format(
                    self.method, self.optimizer, res.status, res.success,
                    res.message)

            # wtest = res.x
            # error = sp.optimize.check_grad(
            #     self.logLoss, self.gradLogLoss, wtest, X, T)
            # print "Check-grad error = {0}".format(error)

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
        return {"n_classes": self.n_classes, "method": self.method,
                "optimizer": self.optimizer, "sound": self.sound,
                "params": self.params}
