#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PlotLoss plots partial losses over the 3-class probability triangle.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# A constant for the smallest positive real.
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
                           'JS'     :Jensen-Shannon divergence
                           'L1'     :L1 distance

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
    elif loss_name == 'L1':
        loss = np.sum(np.abs(np.eye(dim) - q), 1).T

    # ## Average loss
    if loss_name == 'DD':
        meanloss = np.dot(eta.T, np.dot(M.T, bloss))
    elif loss_name != 'JS':
        meanloss = np.dot(eta.T, np.dot(B, loss))

    # A special case: Jensen - Shannon Entropy
    if loss_name == 'JS':
        if not (B is None and M is None and Z is None):
            exit("No weak losses have been defined for JS divergences")
        m = (q + eta) / 2
        d_eta = - np.dot(eta.T, np.log(eta / m))
        d_q = - np.dot(q.T, np.log(q / m))
        meanloss = (d_eta + d_q) / 2

    return meanloss


def compute_simplex(loss_name, V, U, M, Z, eta, N=300):

    # ## Points (q0,q1,q2) to evaluate in the probability triangle
    p = np.linspace(0, 1, N)    # Values of q2;
    delta = np.linspace(-1, 1, N)   # Values of q0-q1;

    MinLossij = 1e10
    meanloss = np.ma.masked_array(np.zeros((N, N)))

    B = np.dot(M.T, np.dot(np.linalg.inv(V.T), U))

    for i in range(N):
        for j in range(N):

            # ## Compute class probabilities corresponding to delta[i], p[j]
            q2 = p[j]
            q1 = (1 - q2 + delta[i]) / 2
            q0 = (1 - q2 - delta[i]) / 2
            q = np.array([q0, q1, q2])

            if np.all(q >= 0):

                # The point is in the probability triange. Evaluate loss
                meanloss[i, j] = compute_loss(q, eta, loss_name, B, M, Z)

                # Locate the position of the minimum loss
                if meanloss[i, j] < MinLossij:
                    MinLossij = meanloss[i, j]
                    destMin = delta[i]
                    pestMin = p[j]

            else:

                # The point is out of the probability simplex.
                # WARNING: surprisingly enough, the command below does not work
                # if I use meanloss[i][j] instead of meanloss[i, j]
                # if loss_name=='log' and i==0 and j==0:
                #     ipdb.set_trace()
                meanloss[i, j] = np.ma.masked

    return meanloss, destMin, pestMin


def main():

    # ## Evaluate loss in the probability triangle

    #########################
    # Configurable parameters

    # Parameters
    eta = np.array([0.35, 0.2, 0.45])       # Location of the minimum
    loss_names = ['square', 'log', 'DD']    # 'square', 'log', 'l01' or 'DD'
    tags = ['Brier', 'CE', 'OSL']
    n_loss = len(loss_names)

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
    d, c = M.shape

    # Parameters for the plots
    fs = 12   # Font size

    # Other common parameters
    # ## Location of the  minimum
    pMin = eta[2]
    dMin = eta[1] - eta[0]

    # ## Points (q0, q1, q2) to evaluate in the probability triangle
    N = 300
    p = np.linspace(0, 1, N)        # Values of q2;
    delta = np.linspace(-1, 1, N)   # Values of q0-q1;

    # This is a scale factor to make sure that the plotted triangle is
    # equilateral
    alpha = 1 / np.sqrt(3)

    # Saturated values. These values have been tested experimentally:
    vmax = {'square': 1,
            'log': 1.3,
            'l01': 1,
            'DD': 1.2,
            'JS': 2,
            'L1': 2}

    fig = plt.figure(figsize=(4 * n_loss, 2.6))

    for i, loss_name in enumerate(loss_names):

        print(loss_name)

        # ###################
        # ## Loss computation

        # Compute loss values over the probability simplex
        meanloss, destMin, pestMin = compute_simplex(loss_name, V, U, M, Z,
                                                     eta)

        # ## Paint loss surface
        meanloss[meanloss > vmax[loss_name]] = vmax[loss_name]

        mlMax = np.max(np.max(meanloss))
        mlMin = np.min(np.min(meanloss))
        scaledmeanloss = 0.0 + 1.0 * (meanloss - mlMin) / (mlMax - mlMin)

        # Contour plot
        # from matplotlib import colors,
        ax2 = fig.add_subplot(1, n_loss, i + 1)
        xx, yy = np.meshgrid(p, delta)
        levs = np.linspace(0, vmax[loss_name], 10)
        mask = scaledmeanloss == np.inf

        scaledmeanloss[mask] = np.ma.masked

        ax2.contourf(alpha * yy, xx, scaledmeanloss, levs, cmap=cm.Blues)

        # Plot true posterior
        ax2.scatter(alpha * dMin, pMin, color='g')
        ax2.text(alpha * dMin + 0.025, pMin, '$\eta$', size=fs)

        # Plot minimum (estimated posterior)
        ax2.scatter(alpha * destMin, pestMin, color='k')
        ax2.text(alpha * destMin + 0.025, pestMin, '$\eta^*$', size=fs)
        ax2.plot([-alpha, 0, alpha, -alpha], [0, 1, 0, 0], '-')
        ax2.axis('off')

        # ## Write labels
        ax2.text(-0.74, -0.1, '$(1,0,0)$', size=fs)
        ax2.text(0.49, -0.1, '$(0,1,0)$', size=fs)
        ax2.text(0.05, 0.96, '$(0,0,1)$', size=fs)
        ax2.axis('equal')

        ax2.set_title(tags[i], size=fs)

    plt.show(block=False)

    plt.savefig('example.svg')

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

