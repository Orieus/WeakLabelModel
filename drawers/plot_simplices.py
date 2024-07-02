#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PlotLoss plots partial losses over the 3-class probability triangle.
import numpy as np
import sys
import copy

# WARNING: This may be needed to run on some versions of macOS. Uncomment if
#          necessary
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

# A constant for the smallest positive real.
EPS = np.nextafter(0, 1)


def discrepancy(q, eta, name, B=None, M=None):
    '''
    Computes a discrepancy measure between probability vectors q and eta

    The discrepancy measure can be a loss function, a divergence measure or
    other

    For a loss function the mean value of a loss l(·, q), for probability
    vector q and for a true posterior, eta.

    The loss is computed as

        l(eta) = eta' L(q)

    where L(q) = (l(0, q), l(1, q), ..., l(c-1, q)) where c is the
    dimension of q and eta.

    Parameters
    ----------
    eta: array, shape (c, 1)
        True posterior with dimension c
    q: array, shape (c, 1)
        Estimate of the posterior
    name: str
        Name of the discrepancy measure. Available options are:
        LOSSES:
        - 'square' :Square loss: D = E_{y~eta}{||q - y||^2}
        - 'L1loss' :L1 distance: D = E_{y~eta}{||q - y||_1}
        - 'log'    :Log loss (cross entroy). D = eta' log(q)
        - 'l01'    :zero-one loss. D = E_{y~eta}(y'hardmax(q)}
        - 'DD'     :Decision directed loss (also called OSL)
        DIVERGENCES:
        - 'L2'     :Euclidean distance: ||p - q||^2
        - 'L1'     :L1 distance: ||p - q||_1
        - 'KL'     :Kullback-Leibler
        - 'JS'     :Jensen-Shannon distance,
        OTHER
        - 'dH'     :Absolute difference between entropies
    B: numpy.ndarray or None, optional (default=None)
        Bias matrix. For an M-proper loss, the bias matrix is the product of
        the mixing matrix (M) and the reconstruction matrix (R).
        For M-proper losses, B is identity matrix.
        Not used used for divergences. Not used for loss DD
    M: numpy.ndarray or None, optional (default=None)
        Transition matrix (only for loss 'DD')

    Returns
    -------
    D: float
        Mean loss D(eta, q)
    '''

    if name in {'square', 'L1loss', 'log', 'l01', 'DD'}:
        # ##############
        # LOSS FUNCTIONS

        # Compute the loss values
        if name == 'square':   # ## Square loss
            c = len(q)
            loss = np.sum((np.eye(c) - q)**2, 1).T

        elif name == 'log':     # ## Cross entropy
            loss = np.minimum(-np.log(q + EPS), 10)

        elif name == 'l01':
            c = len(q)
            loss = np.ones(c)
            loss[np.argmax(q)] = 0

        elif name == 'DD':
            # Possible weak labels
            if M.shape[0] == 6:
                Z = np.array([[1, 0, 0, 1, 1, 0],
                              [0, 1, 0, 1, 0, 1],
                              [0, 0, 1, 0, 1, 1]]).astype('float').T
            else:
                Z = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
                              [0, 0, 1, 0, 1, 0, 1, 1],
                              [0, 0, 0, 1, 0, 1, 1, 1]]).astype('float').T

            bloss = - np.log(np.max(Z * q + EPS, axis=1, keepdims=True))

        elif name == 'L1loss':
            c = len(q)
            loss = np.sum(np.abs(np.eye(c) - q), 1).T

        # Compute the mean loss
        if name == 'DD':
            D = eta.T @ (M.T @ bloss)
        else:
            D = eta.T @ (B @ loss)

    elif name in {'L2', 'L1', 'KL', 'JS', 'dH', 'He'}:
        # DIVERGENCES AND OTHER DISCREPANCY MEASURES
        if name == 'L2':
            # Squared euclidean distance
            D = np.sum((eta - q)**2)

        elif name == 'L1':
            # L1 distance
            D = np.sum(np.abs(eta - q))

        elif name == 'KL':
            # KL divergence
            D = - eta.T @ np.log((q + EPS) / eta)

        elif name == 'JS':
            # Jensen-Shannon distance
            m = (q + eta) / 2
            d_eta = eta.T @ np.log(eta / m)
            d_q = q.T @ np.log((q + EPS) / m)
            D = (d_eta + d_q) / 2

        elif name == 'dH':
            # Absolute difference between entropies.
            h_q = - q.T @ np.log(q + EPS)
            h_eta = - eta.T @ np.log(eta + EPS)
            D = np.abs(h_q - h_eta)

        elif name == 'He':
            # Hellinger distance
            # D = np.sum((np.sqrt(eta) - np.sqrt(q))**2)
            D = 2 * (1 - np.sqrt(eta).T @ np.sqrt(q))

    else:
        sys.exit("Unknown name of the discrepancy measure")
    return D


def compute_simplex(name, eta, R=None, M=None, N=300):
    """
    Computes a discrepancy meeasure D(q, eta) between a given eta and all
    values of q in a grid over the triangular simplex of 3-class probabilities

    Parameters
    ----------
    name: str
        Name of the discrepancy measure. Available options are:
        LOSSES:
        - 'square' :Square loss: D = E_{y~eta}{||q - y||^2}
        - 'L1loss' :L1 distance: D = E_{y~eta}{||q - y||_1}
        - 'log'    :Log loss (cross entroy). D = eta' log(q)
        - 'l01'    :zero-one loss. D = E_{y~eta}(y'hardmax(q)}
        - 'DD'     :Decision directed loss (also called OSL)
        DIVERGENCES:
        - 'L2'     :Euclidean distance: ||p - q||^2
        - 'L1'     :L1 distance: ||p - q||_1
        - 'KL'     :Kullback-Leibler
        - 'JS'     :Jensen-Shannon distance,
        OTHER
        - 'dH'     :Absolute difference between entropies
    eta: array, shape (dimension, 1)
        True posterior
    R: np.ndarray (3, n_weak_classes) or None, optional (defautl=None)
        Reconstruction matrix, with as many columns as the number of weak
        classes. Not used for divergences. Not used for DD loss
    M: np.ndarray (n_weak_classes, 3) or None, optional (defautl=None)
        Transition matrix. Not used for divergences
    N: int
        Size of the grid

    Returns
    -------
    meandiv : array (N, N)
        Divergence values
    delta: numpy.ndarray
        Range of delta-coordinates
    p: numpy.ndarray
        Range of p-coordinates
    delta_min : float
        delta-coordinate o the minimizer
    p_min : float
        Second component of the minimizer
    """

    # ## Points (q0,q1,q2) to evaluate in the probability triangle
    p = np.linspace(0, 1, N)    # Values of q2;
    delta = np.linspace(-1, 1, N)   # Values of q0-q1;

    D = np.ma.masked_array(np.zeros((N, N)))

    # Bias matrix, B
    # The bias matrix is computed as the product of the recontruction matrix R
    # and the mixing matrix M. For M-proper losses, B is the identity matrix
    # In other cases, it is not, producing a shift in the location of the
    # minimum discrepancy wrt eta
    if R is not None and M is not None:
        B = (R @ M).T
    else:
        B = None

    # Initial valua of the min loss (a huge number to be updated in the loop)
    MinLossij = 1e10

    for i in range(N):
        for j in range(N):

            # ## Compute class probabilities corresponding to delta[i], p[j]
            q2 = p[j]
            q1 = (1 - q2 + delta[i]) / 2
            q0 = (1 - q2 - delta[i]) / 2
            q = np.array([q0, q1, q2])

            if np.all(q >= 0):

                # The point is in the probability triange. Evaluate loss
                D[i, j] = discrepancy(q, eta, name, B, M)

                # Locate the position of the minimum loss
                if D[i, j] < MinLossij:
                    MinLossij = D[i, j]
                    delta_min = delta[i]
                    p_min = p[j]

            else:

                # The point is out of the probability simplex.
                # WARNING: surprisingly enough, the command below does not work
                # if I use meanloss[i][j] instead of meanloss[i, j]
                D[i, j] = np.ma.masked

    return D, delta, p, delta_min, p_min


def draw_simplex(D, p, delta, ax2, vmax, eta=None, destMin=None, pestMin=None,
                 tag='', fs=12):
    """
    Draws a discrepancy map over the 3-class probability simplex

    Parameters
    ----------
    D: numpy.ndarray
        Matrix of discrepancy values.
    p: numpy.ndarray
        p-coordinates of the grid.
        The p-coordinate of a probability vector (p0, p1, p2) is p2.
    delta: numpy.ndarray
        delta-coordinates of the grid
        The delta-coordinate of a probability vector (p0, p1, p2) is p1-p0.
    ax2: matplotlib axes
        Handle to the figure where the plot will be loccate
    vmax: float
        Maximum value of the discrepancy. Discrepancies above vmax are
        truncated to vmax. This is to avoid oversaturation caused by small
        regions with very large discrepancy values.
    eta: numpy.ndarray or None, optional (default=None)
        Probability vector. It represent the expected location of the minimum
        of the discrepancy map
    destMin: numpy.ndarray or None, optional (default=None)
        delta-coordinate of the minimum of D_scaled
    pestMin: numpy.ndarray or None, optional (default=None)
        p-coordinate of the minimum of D_scaled
    tag: str (default='')
        Title of the plot
    fs: int, optional (default=12)
        Font size
    """

    # Make a copy of the inypt matrix for rescaling values
    D_scaled = copy.copy(D)

    # Truncate matrix values above the upper limint
    D_scaled[D_scaled > vmax] = vmax
    D_max = np.max(D_scaled)
    D_min = np.min(D_scaled)
    D_scaled = 0.0 + 1.0 * (D_scaled - D_min) / (D_max - D_min)

    # I do not remember why this code is here. It seems useless, but I
    # leave it just in case.
    mask = D_scaled == np.inf
    D_scaled[mask] = np.ma.masked

    # This is a scale factor to make sure that the plotted triangle is
    # equilateral
    alpha = 1 / np.sqrt(3)

    # Contour plot
    # from matplotlib import colors,
    xx, yy = np.meshgrid(p, delta)
    levs = np.linspace(0, vmax, 10)

    ax2.contourf(alpha * yy, xx, D_scaled, levs, cmap=cm.Blues)

    # Plot true probabiilty vector
    if eta is not None:
        pMin = eta[2]
        dMin = eta[1] - eta[0]
        ax2.scatter(alpha * dMin, pMin, color='g')
        ax2.text(alpha * dMin + 0.025, pMin, '$\\eta$', size=fs)

    # Plot minimum (estimated posterior)
    if destMin is not None and pestMin is not None:
        ax2.scatter(alpha * destMin, pestMin, color='k')
        ax2.text(alpha * destMin + 0.025, pestMin, '$\\eta^*$', size=fs)
        ax2.plot([-alpha, 0, alpha, -alpha], [0, 1, 0, 0], '-')

    ax2.axis('off')

    # ## Write labels
    ax2.text(-0.74, -0.1, '$(1,0,0)$', size=fs)
    ax2.text(0.49, -0.1, '$(0,1,0)$', size=fs)
    ax2.text(0.05, 0.96, '$(0,0,1)$', size=fs)
    ax2.axis('equal')

    ax2.set_title(tag, size=fs)

    return ax2


def scatter_probs(eta, ax2=None, fs=12):

    if ax2 is None:
        fig = plt.figure(figsize=(8, 5.2))
        ax2 = fig.add_subplot(1, 1, 1)

    # ## Points (q0,q1,q2) to evaluate in the probability triangle
    N = 200
    p = np.linspace(0, 1, N)    # Values of q2;
    delta = np.linspace(-1, 1, N)   # Values of q0-q1;

    D = np.ma.masked_array(np.zeros((N, N)))

    for i in range(N):
        for j in range(N):

            # ## Compute class probabilities corresponding to delta[i], p[j]
            q2 = p[j]
            q1 = (1 - q2 + delta[i]) / 2
            q0 = (1 - q2 - delta[i]) / 2
            q = np.array([q0, q1, q2])

            if np.all(q >= 0):
                # The point is in the probability triange. Evaluate loss
                D[i, j] = 1
            else:
                # The point is out of the probability simplex.
                D[i, j] = np.ma.masked

    # This is a scale factor to make sure that the plotted triangle is
    # equilateral
    alpha = 1 / np.sqrt(3)

    # Contour plot
    # from matplotlib import colors,
    xx, yy = np.meshgrid(p, delta)
    levs = np.linspace(0, 2, 10)
    ax2.contourf(alpha * yy, xx, D, levs, cmap=cm.Blues)
    border_delta = np.array([-1, 1, 0, -1])
    border_p = np.array([0, 0, 1, 0])
    ax2.plot(alpha * border_delta, border_p, color='black')

    # Plot true probabiilty vector
    p = eta[:, 2]
    delta = eta[:, 1] - eta[:, 0]
    ax2.scatter(alpha * delta, p, color='g', s=5)
    ax2.axis('off')

    # ## Write labels
    ax2.text(-0.74, -0.1, '$(1,0,0)$', size=fs)
    ax2.text(0.49, -0.1, '$(0,1,0)$', size=fs)
    ax2.text(0.05, 0.96, '$(0,0,1)$', size=fs)
    ax2.axis('equal')

    return ax2


def compute_draw_simplices(D_names, tags, eta, vmax=None, R=None, M=None,
                           N=300, fs=12):
    """
    Computes and draws a bunch of discrepancy maps over the 3-class probability
    simplex

    Parameters
    ----------
    D_names: list of str
        Names of the discrepancies to plot. Available options are:
        LOSSES:
        - 'square' :Square loss: D = E_{y~eta}{||q - y||^2}
        - 'L1loss' :L1 distance: D = E_{y~eta}{||q - y||_1}
        - 'log'    :Log loss (cross entroy). D = eta' log(q)
        - 'l01'    :zero-one loss. D = E_{y~eta}(y'hardmax(q)}
        - 'DD'     :Decision directed loss (also called OSL)
        DIVERGENCES:
        - 'L2'     :Euclidean distance: ||p - q||^2
        - 'L1'     :L1 distance: ||p - q||_1
        - 'KL'     :Kullback-Leibler
        - 'JS'     :Jensen-Shannon distance,
        OTHER
        - 'dH'     :Absolute difference between entropies
    tag: list of str
        Titles of the plots, one per discrepancy.
    vmax: list of float or None
        List of maximum values of the discrepancies. Discrepancies above vmax
        are truncated to vmax. This is to avoid oversaturation caused by small
        regions with very large discrepancy values.
        If None, a default value 1.6 is used
    eta: numpy.ndarray or None, optional (default=None)
        Probability vector. It represents the expected location of the minimum
        of the discrepancy map
    R: np.ndarray (3, n_weak_classes) or None, optional (defautl=None)
        Reconstruction matrix, with as many columns as the number of weak
        classes. Not used for divergences. Not used for DD loss
    M: np.ndarray (n_weak_classes, 3) or None, optional (defautl=None)
        Transition matrix. Not used for divergences
    N: int
        Size of the grid
    fs: int, optional (default=12)
        Font size
    """

    n_plots = len(D_names)

    # Default vmax
    if vmax is None:
        vmax = [1.6] * n_plots


    fig = plt.figure(figsize=(4 * n_plots, 2.6))

    for i, name in enumerate(D_names):

        print(name)
        tag_i = tags[i]
        vmax_i = vmax[i]

        # ###################
        # ## Loss computation

        # Compute loss values over the probability simplex
        D, delta, p, delta_min, p_min = compute_simplex(name, eta, R, M, N)

        ax2 = fig.add_subplot(1, n_plots, i + 1)
        ax2 = draw_simplex(D, p, delta, ax2, vmax_i, eta, delta_min, p_min,
                           tag_i, fs)

    return fig


def main():

    # ## Evaluate loss in the probability triangle

    #########################
    # Configurable parameters

    # Parameters
    eta = np.array([0.35, 0.2, 0.45])       # Location of the minimum
    eta = np.array([0.8, 0.1, 0.1])       # Location of the minimum
    loss_names = ['square', 'log', 'DD']    # 'square', 'log', 'l01' or 'DD'
    tags = ['Brier', 'CE', 'OSL']

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
    M = np.array([[0.6, 0.0, 0.0, 0.2, 0.2, 0],
                  [0.0, 0.8, 0.0, 0.1, 0.0, 0.1],
                  [0.0, 0.0, 0.4, 0.0, 0.3, 0.3]]).T

    # Parameters for the plots
    fs = 12   # Font size

    # ## Points (q0, q1, q2) to evaluate in the probability triangle
    N = 300

    # Saturated values. These values have been tested experimentally:
    vmax_all = {'square': 1,
                'log': 1.6,
                'l01': 1,
                'DD': 1.2,
                'JS': 2,
                'L1loss': 2,
                'L1': 1.6,
                'L2': 1.6,
                'KL': 1.6,
                'JS': 1.6,
                'dH': 1.6,
                'He': 1.6}

    vmax = [vmax_all[name] for name in loss_names]
    compute_draw_simplices(loss_names, tags, eta, vmax, R, M, N, fs)

    plt.show(block=False)
    # breakpoint()

    fname = 'example.svg'
    plt.savefig(fname)
    print(f"Figure saved in {fname}")

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

