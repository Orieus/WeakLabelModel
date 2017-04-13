#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Defines classifier objects that work with weak labels

    Author: Miquel Perello-Nieto, Apr 2017
"""
import numpy as np

import theano
import theano.tensor as T

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Nadam
from functools import partial

theano.config.exception_verbosity = 'high'


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


def brier_loss(y_true, y_pred):
    """ Computes brier score for the given tensors

    equivalent to:
            w = class_weigths
            N, C = y_true.shape
            bs = 0
            for n in range(N):
                for c in range(C):
                    bs += w[c]*(y_pred[n, c] - y_true[n, c])**2
            return bs/N
    """
    return T.mean(T.square(T.sub(y_pred, y_true)), axis=-1)


# FIXME add the parameter rho to the gradient descent
class KerasModel(object):
    def __init__(self, input_size, output_size, optimizer='SGD',
                 batch_size=None, class_weights=None, params={}):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.params = params
        self.optimizer = optimizer

        model = self.create_model(input_size, output_size)

        if class_weights is None:
            class_weights = np.ones(output_size)
        self.class_weights = class_weights

        #wbl = partial(w_brier_loss, class_weights=class_weights)
        #wbl.__name__ = 'w_brier_loss'
        wbl = brier_loss
        wbl.__name__ = 'brier_loss'

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

        model.compile(loss=wbl, optimizer=keras_opt, metrics=['acc'])

        self.model = model

    def create_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(output_size, input_shape=(input_size,)))
        model.add(Activation('softmax'))
        return model

    def fit(self, train_x, train_y, test_x=None, test_y=None, batch_size=None,
            nb_epoch=1):
        """
        The fit function requires both train_x and train_y.
        See 'The selected model' section above for explanation
        """
        if 'n_it' in self.params:
            nb_epoch = self.params['n_it']

        batch_size = self.batch_size if batch_size is None else batch_size

        if batch_size is None:
            batch_size = train_x.shape[0]

        return self.model.fit(train_x, train_y, batch_size=batch_size,
                              nb_epoch=nb_epoch, verbose=0)

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
                "class_weights": self.class_weights, "params": self.params}


class KerasWeakLogisticRegression(KerasModel):
    def create_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(output_size, input_shape=(input_size,)))
        model.add(Activation('softmax'))
        return model


class KerasWeakMultilayerPerceptron(KerasModel):
    def create_model(self, input_size, output_size):
        model = Sequential()
        model.add(Dense(200, input_shape=(input_size,)))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size))
        model.add(Activation('softmax'))
        return model
