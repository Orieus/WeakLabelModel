#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PlotLoss plots partial losses over the 3-class probability triangle.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ipdb

# Parameters
eta = np.array([0.45, 0.15, 0.4])  # Location of the minimum
N = 200                  # No. of points for the 2D plot
fs = 24                  # Fontsize for the plots
LossName = 'log'         # 'square', 'log' or 'l01'
# LossName = 'square'    # 'square', 'log' or 'l01'


# ## Ambiguity set
#    001 010 100 011 101 110
V = np.array([[1, 0, 0, 0,   0,   0],
              [0, 0, 0, 0.5, 0.5, 0],
              [0, 1, 0, 0,   0,   0],
              [0, 0, 0, 0.5, 0,   0.5],
              [0, 0, 1, 0,   0,   0],
              [0, 0, 0, 0,   0.5, 0.5]]).T

U = np.array([[1, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1]]).T

# ## True mixing matrix
#    001 010 100 011 101 110
M = np.array([[0.5, 0,   0,   0.4, 0.1, 0],
              [0,   0.5, 0,   0.3, 0,   0.2],
              [0,   0,   0.6, 0,   0.2, 0.2]]).T

d, c = M.shape
# R0 = 0.1*rand(d,c);
# R1 = 0.1*rand(d,c);
# M  = R1 + (1-R0-R1).*M;
# M  = M./(ones(d,1)*sum(M));

A = np.dot(np.linalg.inv(V.T), U)
B = np.dot(M.T, np.dot(np.linalg.inv(V.T), U))

# ## Points (q0,q1,q2) to evaluate in the probability triangle
p = np.linspace(0, 1, N)    # Values of q2;
delta = np.linspace(-1, 1, N)   # Values of q0-q1;

# ## Evaluate loss in the probability triangle
MinLossij = 1e10
meanloss = np.zeros((N, N))

for i in range(N):
    for j in range(N):

        # ## Compute class probabilities corresponding to delta[i],p[j]
        q2 = p[j]
        q1 = (1 - q2 + delta[i])/2
        q0 = (1 - q2 - delta[i])/2
        q = np.array([q0, q1, q2])

        if np.all(q >= 0):

            # The point is in the probability triange. Evaluate loss
            if LossName == 'square':   # ## Square loss
                loss = np.sum((np.eye(3) - q*np.ones((1, 3)))**2, 1).T
            elif LossName == 'log':     # ## Cross entropy
                # ipdb.set_trace()
                loss = np.minimum(-np.log(q + np.nextafter(0, 1)), 10)
            elif LossName == 'l01':
                loss = float(q == np.max(q))

            # ## Average loss
            meanloss[i][j] = np.dot(eta.T, np.dot(B, loss))
            if meanloss[i][j] < MinLossij:
                MinLossij = meanloss[i][j]
                destMin = delta[i]
                pestMin = p[j]

            if np.isnan(meanloss[i][j]):
                ipdb.set_trace()

        else:
            # The point is out of the probability simplex.
            meanloss[i][j] = None
            ipdb.set_trace()

# ## Locate minimum
pMin = eta[2]
dMin = eta[1]-eta[0]
iMin = np.argmin(np.abs(delta - dMin))
jMin = np.argmin(np.abs(p - pMin))

alpha = 1/np.sqrt(3)

# ## Paint loss surface
mlMax = np.max(np.max(meanloss))
mlMin = np.min(np.min(meanloss))
scaledmeanloss = 0.4 + 0.6 * (meanloss - mlMin) / (mlMax - mlMin)
scaledmeanloss[iMin][jMin] = 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx, yy = np.meshgrid(p, alpha * delta)
ax.plot_surface(xx, yy, scaledmeanloss)

ipdb.set_trace()
# h = plt.contour(p, alpha * delta, scaledmeanloss, 15, 'k')
# colormap gray
# shading interp
# plt.axis('equal')
# plt.view(90, -90)

# ## Plot
# h2 = plt.plot3(np.array([0, 0, 1, 0]),
#                np.array([alpha, -alpha, 0, alpha]),
#                -2 * np.array([1, 1, 1, 1]), 'k')
# set(h2, 'LineWidth', 2)

# # ## Plot true posterior
# h = plt.scatter(pMin, alpha * dMin, -2, color='k', marker='.',
#                 markersize=18)

# # ## Plot estimated posterior
# h = plt.scatter(pestMin, alpha * destMin, -2, color='k', marker='.',
#                 markersize=18)

# # axis off

# # ## Write labels
# plt.text(0, -0.66, 0, '{\bf e}_0', 'FontName', 'Times New Roman', 'FontSize',
#          fs)
# plt.text(0, 0.59, 0, '{\bf e}_1', 'FontName', 'Times New Roman', 'FontSize',
#          fs)
# plt.text(0.98, 0.03,  0, '{\bf e}_2', 'FontName', 'Times New Roman',
#          'FontSize', fs)
# plt.text(pMin, alpha * dMin + 0.04,  0, '\eta', 'FontName', 'Times New Roman',
#          'FontSize', fs)
# plt.text(pestMin, alpha * destMin + 0.04, 0, '\eta^*', 'FontName',
#          'Times New Roman', 'FontSize', fs)
# plt.text(pestMin + 0.02, alpha * destMin + 0.04, 0, '^', 'FontName',
#          'Times New Roman', 'FontSize', fs)

# if LossName == 'square':
#     hgsave('SquareLoss')
#     print -dpdf SquareLoss
#     print -deps SquareLoss
# elif LossName == 'log':
#     hgsave('LogLoss')
#     print -dpdf LogLoss
#     print -deps LogLoss
