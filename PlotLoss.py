#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PlotLoss plots partial losses over the 3-class probability triangle.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import ipdb

EPS = np.nextafter(0, 1)


def compute_loss(q, eta, loss_name, B=None, M=None, Z=None):

    ''' Computes a mean value of the conditional loss l(eta, q), where eta is
        the true posterior and q is an estimate.

        Args:
            eta       : Column vector.
            q         : Column vector with the same dimension than eta
            loss_name : Loss function. Available options are
                           'square' :Square loss
                           'log'    :Log loss (cross entroy)
                           'l01'    :zero-one loss
                           'DD'     :Decision directed loss (also called OSL)

            The following parameters are used to compute weak losses
            B
            M
            Z
    '''

    if loss_name == 'square':   # ## Square loss
        dim = len(q)
        loss = np.sum((np.eye(dim) - q)**2, 1).T
    elif loss_name == 'log':     # ## Cross entropy
        # ipdb.set_trace()
        loss = np.minimum(-np.log(q + EPS), 10)
    elif loss_name == 'l01':
        loss = np.ones(dim)
        loss[np.argmax(q)] = 0
    elif loss_name == 'DD':
        bloss = - np.log(np.max(Z * q + EPS, axis=1, keepdims=True))

    # ## Average loss
    if loss_name == 'DD':
        meanloss = np.dot(eta.T, np.dot(M.T, bloss))
    else:
        meanloss = np.dot(eta.T, np.dot(B, loss))

    return meanloss


def compute_simplex(loss_name, V, U, M, Z, eta, N=300):

    # ## Points (q0,q1,q2) to evaluate in the probability triangle
    p = np.linspace(0, 1, N)    # Values of q2;
    delta = np.linspace(-1, 1, N)   # Values of q0-q1;

    MinLossij = 1e10
    meanloss = np.zeros((N, N))

    B = np.dot(M.T, np.dot(np.linalg.inv(V.T), U))

    for i in range(N):
        for j in range(N):

            # ## Compute class probabilities corresponding to delta[i], p[j]
            q2 = p[j]
            q1 = (1 - q2 + delta[i])/2
            q0 = (1 - q2 - delta[i])/2
            q = np.array([q0, q1, q2])

            if np.all(q >= 0):

                # The point is in the probability triange. Evaluate loss
                meanloss[i][j] = compute_loss(q, eta, loss_name, B, M, Z)

                # Locate the position of the minimum loss
                if meanloss[i][j] < MinLossij:
                    MinLossij = meanloss[i][j]
                    destMin = delta[i]
                    pestMin = p[j]

            else:

                # The point is out of the probability simplex.
                meanloss[i][j] = np.inf

    return meanloss, destMin, pestMin


def main():

    # ## Evaluate loss in the probability triangle

    #########################
    # Configurable parameters

    # Parameters
    eta = np.array([0.35, 0.2, 0.45])  # Location of the minimum
    loss_name = 'DD'         # 'square', 'log', 'l01' or 'DD'

    # ## Possible weak labels
    Z = np.array([[1.0, 0, 0, 1, 1, 0],
                  [0,   1, 0, 1, 0, 1],
                  [0,   0, 1, 0, 1, 1]]).T

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
    # M = np.array([[0.5, 0,   0,   0.4, 0.1, 0],
    #               [0,   0.5, 0,   0.3, 0,   0.2],
    #               [0,   0,   0.6, 0,   0.2, 0.2]]).T
    M = np.array([[0.6, 0,   0,   0.2, 0.2, 0],
                  [0,   0.8, 0,   0.1, 0,   0.1],
                  [0,   0,   0.4, 0,   0.3, 0.3]]).T

    # Parameters for the plots
    fs = 14   # Font size

    # ###################
    # ## Loss computation

    # Compute loss values over the probability simplex
    d, c = M.shape
    meanloss, destMin, pestMin = compute_simplex(loss_name, V, U, M, Z, eta)

    # ## Locate minimum
    pMin = eta[2]
    dMin = eta[1] - eta[0]

    # ## Points (q0, q1, q2) to evaluate in the probability triangle
    N = 300
    p = np.linspace(0, 1, N)        # Values of q2;
    delta = np.linspace(-1, 1, N)   # Values of q0-q1;

    # ## Paint loss surface
    meanloss = np.minimum(meanloss, 2)
    mlMax = np.max(np.max(meanloss))
    mlMin = np.min(np.min(meanloss))
    scaledmeanloss = 0.0 + 1.0 * (meanloss - mlMin) / (mlMax - mlMin)

    # This is a scale factor to make sure that the plotted triangle is
    # equilateral
    alpha = 1 / np.sqrt(3)

    # Contour plot
    # from matplotlib import colors,
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    xx, yy = np.meshgrid(p, delta)
    levs = np.linspace(0, 1, 50)
    mask = scaledmeanloss == np.inf
    scaledmeanloss[mask] = np.ma.masked

    ax2.contourf(alpha * yy, xx, scaledmeanloss, levs, cmap=cm.hsv)

    # Plot true posterior
    ax2.scatter(alpha * dMin, pMin, color='g')
    ax2.text(alpha * dMin + 0.025, pMin, '$\eta$', size=fs)

    # Plot minimum (estimated posterior)
    ax2.scatter(alpha * destMin, pestMin, color='k')
    ax2.text(alpha * destMin + 0.025, pestMin, '$\eta^*$', size=fs)
    ax2.plot([-alpha, 0, alpha, -alpha], [0, 1, 0, 0], '-')
    ax2.axis('off')

    # ## Write labels
    ax2.text(-0.74, 0, '$(1,0,0)$', size=fs)
    ax2.text(0.59, 0, '$(0,1,0)$', size=fs)
    ax2.text(0.05, 0.96, '$(0,0,1)$', size=fs)
    # ax2.axis('off')
    ax2.axis('equal')

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

    plt.show(block=False)
    plt.savefig('example.png')

    ipdb.set_trace()


    # if LossName == 'square':
    #     hgsave('SquareLoss')
    #     print -dpdf SquareLoss
    #     print -deps SquareLoss
    # elif LossName == 'log':
    #     hgsave('LogLoss')
    #     print -dpdf LogLoss
    #     print -deps LogLoss

if __name__ == "__main__":
    main()

