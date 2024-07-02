#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PlotLoss plots partial losses over the 3-class probability triangle.
import numpy as np
import plot_simplices as drw

# WARNING: This may be needed to run on some versions of macOS. Uncomment if
#          necessary
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# A constant for the smallest positive real.
EPS = np.nextafter(0, 1)


# ## Evaluate loss in the probability triangle

#########################
# Configurable parameters

# Parameters
eta = np.array([0.35, 0.2, 0.45])       # Location of the minimum
eta = np.array([0.8, 0.1, 0.1])       # Location of the minimum
loss_names = ['log', 'DD']    # 'square', 'log', 'l01' or 'DD'
tags = ['CE', 'OSL']

# Build a reconstruction matrix
V = np.array([[1, 0, 0, 0.0, 0.0, 0],
              [0, 0, 0, 0.5, 0.5, 0],
              [0, 1, 0, 0.0, 0.0, 0],
              [0, 0, 0, 0.5, 0.0, 0.5],
              [0, 0, 1, 0.0, 0.0, 0],
              [0, 0, 0, 0.0, 0.5, 0.5]]).T
U = np.array([[1, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1]]).T
R = U.T @ np.linalg.inv(V)

# Transition matrix
M = np.array([[0.2, 0.0, 0.0, 0.8, 0.0, 0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T

# Weak label probabilities:
p = M @ eta
print(f"-- Weak label probabilities: {p}")
q = np.array([p[0] + p[3] + p[4],
              p[1] + p[3] + p[5],
              p[2] + p[4] + p[5]])
print(f"-- Weak class probabilities: {q}")


# Parameters for the plots
fs = 12   # Font size

# ## Points (q0, q1, q2) to evaluate in the probability triangle
N = 300

# Saturated values. These values have been tested experimentally:
vmax_all = {'square': 1,
            'log': 1.4,
            'l01': 1,
            'DD': 1.4,
            'JS': 2,
            'L1loss': 2,
            'L1': 1.6,
            'L2': 1.6,
            'KL': 1.6,
            'JS': 1.6,
            'dH': 1.6,
            'He': 1.6}

vmax = [vmax_all[name] for name in loss_names]
drw.compute_draw_simplices(loss_names, tags, eta, vmax, R, M, N, fs)

plt.show(block=False)

fname = 'example.svg'
plt.savefig(fname)
print(f"Figure saved in {fname}")



