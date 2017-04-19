#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This code tests the weighted brier score with keras

    Author: MPN, April, 2017
"""

# External modules
import numpy as np
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD, Nadam

import theano.tensor as T

from functools import partial

import matplotlib.pyplot as plt

from testUtils import newfig, plot_data_predictions, get_grid

import pandas as pd

plt.ion()
np.random.seed(42)

def w_brier_loss(o, f, class_w):
    """f is the forecast and o is the original outcome"""
    print class_w
    return T.mean(T.dot(T.square(T.sub(f, o)), class_w), axis=-1)

def brier_score(o, f, class_w, per_class=False):
    if per_class:
        return np.squeeze(np.multiply(
            np.square(o - f).mean(axis=0).reshape(-1, 1),
            class_w.reshape(-1,1)))
    return np.square(o - f).dot(class_w).mean()

n_samples = 200
n_sim = 20
mean1 = [0, 0]
mean2 = [1, 1]
class_weight_ratios = np.power(10.0, np.arange(-5, 5))

epochs = 100

results = []

for simulation in range(n_sim):
    X = np.vstack((np.random.normal(loc=mean1, scale=1.0, size=(n_samples/2,2)),
                   np.random.normal(loc=mean2, scale=1.0, size=(n_samples/2,2))))
    y = np.concatenate((np.zeros(n_samples/2), np.ones(n_samples/2))).astype(int)
    Y = np.vstack((1-y, y)).T

    for class_w_ratio in class_weight_ratios:
        class_weights = [class_w_ratio, 1.0]
        print("Weights ratio = {}".format(class_weights))

        ### Model specification
        model = Sequential()
        model.add(Dense(2, input_shape=(2,), activation='softmax'))

        loss = partial(w_brier_loss, class_w=class_weights)
        loss.__name__ = 'w_brier_loss'

        #optimizer = SGD(lr=0.01, momentum=0.1, decay=0.1, nesterov=False)
        optimizer = Nadam()

        model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

        ### Training
        model.fit(X, Y, epochs=epochs, verbose=0)

        prediction = model.predict(X)
        #bs = np.mean(np.sum(np.square(prediction - Y), axis=1))

        bs = brier_score(Y, prediction, class_w=class_weights)
        print("Brier score = {}".format(bs))
        cm = confusion_matrix(y, np.argmax(prediction, axis=1))
        print("Confusion matrix\n{}".format(cm))
        acc = np.true_divide(np.sum(np.diag(cm)), np.sum(cm))
        print("Accuracy = {}".format(acc))
        sensitivity, specificity = np.true_divide(np.diag(cm), cm.sum(axis=1))
        print("Sensitivity = {}".format(sensitivity))
        print("Specificity = {}".format(specificity))

        results.append([simulation, class_w_ratio, bs, acc, sensitivity,
                        specificity])

#x_grid, MX1, MX2 = get_grid(X, delta=0.1)
#Z = model.predict(x_grid)
#Z = Z[:,0].reshape(MX1.shape[0], MX1.shape[1])
#
#fig = newfig('contour')
#plot_data_predictions(fig, X, y, Z, MX1, MX2)

df = pd.DataFrame(results, columns=['simulation', 'weights_ratio',
                                    'brier_score', 'accuracy', 'sensitivity',
                                    'specificity'])

groupby = 'weights_ratio'
columns = ['accuracy', 'sensitivity', 'specificity']

grouped = df.groupby([groupby])

fig = newfig('class_w')
ax = fig.add_subplot(111)
grouped[columns].mean().plot(style='o-', logx=True, ax=ax)
