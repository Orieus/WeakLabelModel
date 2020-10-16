#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Defines classifier objects that work with weak labels

    Author: Miquel Perello-Nieto, Apr 2017
"""
import numpy as np
import scipy as sp

import theano
import theano.tensor as T
from keras import backend as K


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Nadam
from functools import partial

from sklearn.base import BaseEstimator

theano.config.exception_verbosity = 'high'


def log_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    out = -y_true*K.log(y_pred)
    return K.mean(out, axis=-1)


def w_brier_loss(y_true, y_pred, class_weights):
    """ Computes weighted brier score for the given tensors

    equivalent to:
            w = class_weigths
            N, C = y_true.shape
            bs = 0
            for n in range(N):
                for c in range(C):
                    bs += w[c]*(y_pred[n, c] - y_true[n, c])**2
            return bs/N
    """
    return T.mean(T.dot(T.square(T.sub(y_pred, y_true)), class_weights),
                  axis=-1)


def osl_w_brier_loss(o, f, class_weights):
    """f is the forecast and o is the original outcome"""
    # FIXME argmax does not have keep_dims anymore, find how to make a binary
    # matrix with a one in the highest value of the original matrix
    d = K.argmax(o * f, axis=-1, keep_dims=True)
    return K.mean(K.dot(K.square(f - d), class_weights), axis=-1)


def osl_brier_loss(o, f):
    """f is the forecast and o is the original outcome"""
    d = K.argmax(o * f, axis=-1, keep_dims=True)
    return brier_loss(d, f)


def brier_loss(y_true, y_pred):
    """ Computes weighted brier score for the given tensors

    equivalent to:
            w = class_weigths
            N, C = y_true.shape
            bs = 0
            for n in range(N):
                for c in range(C):
                    bs += (y_pred[n, c] - y_true[n, c])**2
            return bs/N
    """
    return K.mean(K.sum(K.square(y_pred - K.cast(y_true, K.floatx())), axis=1))


# FIXME add the parameter rho to the gradient descent
class KerasModel(BaseEstimator):
    def __init__(self, input_size, output_size, optimizer='SGD',
                 batch_size=None, class_weights=None, OSL=False, params={},
                 random_seed=None):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.params = params
        self.optimizer = optimizer
        self.OSL = OSL
        if 'random_seed' in params.keys():
            random_seed = params['random_seed']
        self.random_seed = random_seed

        # TODO see why I can not initialize the seed just before I call compile
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        model = self.create_model(input_size, output_size)

        if class_weights is None:
            self.class_weights = np.ones(output_size)
        else:
            self.class_weights = class_weights

        if OSL is True:
            ## FIXME There is a problem with argmax() and argument keep_dims
            if np.all(self.class_weights):
                loss = osl_brier_loss
                loss.__name__ = 'osl_brier_loss'
            else:
                loss = partial(osl_w_brier_loss, class_weights=self.class_weights)
                loss.__name__ = 'osl_w_brier_loss'
        else:
            if np.all(self.class_weights):
                loss = brier_loss
                loss.__name__ = 'brier_loss'
            else:
                loss = partial(w_brier_loss, class_weights=self.class_weights)
                loss.__name__ = 'w_brier_loss'

        # FIXME adjust the parameter rho
        if 'rho' in self.params:
            lr = self.params['rho']
        elif optimizer == 'SGD':
            lr = 1.0
        elif optimizer == 'Adam':
            lr = 0.001
        elif optimizer == 'Nadam':
            lr = 0.002

        if optimizer == 'SGD':
            keras_opt = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        elif optimizer == 'Adam':
            keras_opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                             decay=0.0)
        elif optimizer == 'Nadam':
            keras_opt = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                              schedule_decay=0.004)
        else:
            raise('Optimizer unknown: {}'.format(optimizer))

        model.compile(loss=loss, optimizer=keras_opt, metrics=['acc'])

        self.model = model

    def create_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(output_size, input_shape=(input_size,)))
        model.add(Activation('softmax'))
        return model

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

    def fit(self, train_x, train_y, test_x=None, test_y=None, batch_size=None,
            epochs=1):
        """
        The fit function requires both train_x and train_y.
        See 'The selected model' section above for explanation
        """
        if 'n_epoch' in self.params:
            epochs = self.params['n_epoch']

        batch_size = self.batch_size if batch_size is None else batch_size

        if batch_size is None:
            batch_size = train_x.shape[0]

        # TODO try to use the OSL loss instead of iterating over epochs
        if self.OSL:
            history = []
            for n in range(epochs):
                predictions = self.model.predict_proba(train_x,
                                                       batch_size=batch_size,
                                                       verbose=0)
                train_osl_y = self.hardmax(np.multiply(train_y, predictions))

                h = self.model.fit(train_x, train_osl_y, batch_size=batch_size,
                                   epochs=1, verbose=0)
                history.append(h)
            return history

        return self.model.fit(train_x, train_y, batch_size=batch_size,
                              epochs=epochs, verbose=0)

    def predict(self, X, batch_size=None):
        # Compute posterior probability of class 1 for weights w.
        p = self.predict_proba(X, batch_size=batch_size)

        # Class
        D = np.argmax(p, axis=1)

        return D  # p, D

    def predict_proba(self, test_x, batch_size=None):
        """
        This function finds the k closest instances to the unseen test data,
        and averages the train_labels of the closest instances.
        """
        batch_size = self.batch_size if batch_size is None else batch_size

        if batch_size is None:
            batch_size = test_x.shape[0]

        return self.model.predict(test_x, batch_size=batch_size)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"input_size": self.input_size, "output_size": self.output_size,
                "optimizer": self.optimizer, "batch_size": self.batch_size,
                "class_weights": self.class_weights, "params": self.params,
                "OSL": self.OSL, "random_seed": self.random_seed}


class KerasWeakLogisticRegression(KerasModel):
    def create_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(output_size, input_shape=(input_size,),
                        kernel_initializer='glorot_uniform'))
        model.add(Activation('softmax'))
        return model


class KerasWeakMultilayerPerceptron(KerasModel):
    def create_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(200, input_shape=(input_size,), kernel_initializer='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(200, kernel_initializer='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size))
        model.add(Activation('softmax'))
        return model
