#!/usr/bin/env python
# -*- coding: utf-8 -*-

# External modules
import numpy as np
# from numpy import binary_repr
import sklearn.datasets as skd           # Needs version 0.14 or higher
# import sklearn.linear_model as sklm
import sys
import ipdb


def computeM(c, alpha=0.5, beta=0.5, gamma=0.5, method='supervised'):
    """
    Generate a mixing matrix M, given the number of classes c.
    """

    if method == 'supervised':

        M = np.array([[0.0, 0.0, 0.0],
                      [0,   0,   1],
                      [0,   1,   0],
                      [0.0, 0.0, 0.0],
                      [1,   0,   0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])

    elif method == 'noisy':

        M = np.array([[0.0,     0.0,    0.0],
                      [alpha/2, beta/2, 1-gamma],
                      [alpha/2, 1-beta, gamma/2],
                      [0.0,     0.0,    0.0],
                      [1-alpha, beta/2, gamma/2],
                      [0.0,     0.0,    0.0],
                      [0.0,     0.0,    0.0],
                      [0.0,     0.0,    0.0]])

    elif method == 'IPL':

        M = np.array([
                [0.0,             0.0,           0.0],
                [0,               0,             (1-gamma)**2],
                [0,               (1-beta)**2,   0],
                [0.0,             beta*(1-beta), gamma*(1-gamma)],
                [(1-alpha)**2,    0,             0],
                [alpha*(1-alpha), 0.0,           gamma*(1-gamma)],
                [alpha*(1-alpha), beta*(1-beta), 0.0],
                [alpha**2,        beta**2,       gamma**2]])

    elif method == 'IPL_old':

        M = np.array([
                [0.0,              0.0,            0.0],
                [0,                0,              1-gamma-gamma**2],
                [0,                1-beta-beta**2, 0],
                [0.0,              beta/2,         gamma/2],
                [1-alpha-alpha**2, 0,              0],
                [alpha/2,          0.0,            gamma/2],
                [alpha/2,          beta/2,         0.0],
                [alpha**2,         beta**2,        gamma**2]])

    elif method == 'quasi_IPL':

        # Convert beta to numpy array
        if isinstance(beta, (list, tuple, np.ndarray)):
            # Make sure beta is a numpy array
            beta = np.array(beta)
        else:
            beta = np.array([beta] * c)

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(1, d-1):

            z_bin = [int(b) for b in bin(z)[2:].zfill(c)]
            modz = sum(z_bin)

            M[z, :] = z_bin*(beta**(modz-1) * (1-beta)**(c-modz))

        M = M / np.sum(M, axis=0)

    else:

        print "Error: unknown method"
        sys.exit()

    return M


def generateWeak(y, M, c):
    """
   Generate the set of weak labels z for n examples, given the ground truth
   labels y for n examples, a mixing matrix M, and a number of classes c.
    """

    z = np.zeros(y.shape)           # Weak labels for all labels y (int)
    d = 2 ** c                      # Number of weak labels
    dec_labels = np.arange(d)       # Possible weak labels (int)

    for index, i in enumerate(y):
        z[index] = np.random.choice(dec_labels, 1, p=M[:, i])

    return z


def computeVirtual(z, c, method='IPL', M=None):
    """
    Generate the set of virtual labels v for n examples, given the weak labels
    for n examples in decimal format, a mixing matrix M, the number of classes
    c, and a method.

    Args:
        z       :List of weak labels. Each weak label is an integer whose
                 binary representation encondes the observed weak labels
        c       :Number of classes. All components of z must be smaller than
                 2**c
        method  :Method applied to compute the virtual label vector v.
                 Available methods are:
                 - 'IPL'  : Independet Partial Labels. Takes virtual label
                            vectors equal to the binary representations of the
                            the weak labels in z
                 - 'supervised': Equivalent to IPL
                 - 'Mproper'   : Computes virtual labels for a M-proper loss.
                 - 'MCC'       : Computes virtual labels for a M-CC loss
                                 (Not available yet)
        M       :Mixing matrix. Only for methods 'Mproper' and 'MCC'

    Returns:
        v
    """

    z_bin = np.zeros((z.size, c))         # weak labels (binary)
    v = np.zeros((z.size, c))             # virtual labels

    for index, i in enumerate(z):         # From dec to bin

        z_bin[index, :] = [int(x) for x in bin(int(i))[2:].zfill(c)]

    if method == 'IPL' or method == 'supervised':

        # weak and virtual are the same
        pass

    elif method == 'quasi_IPL':    # quasi-independent labels

        for index, i in enumerate(z_bin):

            aux = z_bin[index, :]
            weak_pos = np.sum(aux)

            if not weak_pos == c:

                weak_zero = float(1-weak_pos)/(c-weak_pos)
                aux[aux == 0] = weak_zero
                z_bin[index, :] = aux

            else:

                z_bin[index, :] = np.array([None] * c)

    elif method == 'Mproper':

        # Compute the virtual label matrix
        Y = np.linalg.pinv(M)

        # Compute the virtual label.
        for index, i in enumerate(z):
            # The virtual label for weak label i is the i-th row in Y
            z_bin[index, :] = Y[:, int(i)]

    else:

        print 'Unknown method. Weak label taken as virtual'

    v = z_bin

    return v


def main():

    # #########################################################################
    # ## MAIN #################################################################
    # #########################################################################

    ############################
    # ## Configurable parameters

    # Parameters for sklearn synthetic data
    ns = 100    # Sample size
    nf = 2      # Data dimension
    c = 3       # Number of classes

    #####################
    # ## A title to start

    print "======================="
    print "    Weak labels"
    print "======================="

    ###########################################################################
    # ## PART I: Load data (samples and true labels)                         ##
    ###########################################################################

    X, y = skd.make_classification(
        n_samples=ns, n_features=nf, n_informative=2, n_redundant=0,
        n_repeated=0, n_classes=c, n_clusters_per_class=1, weights=None,
        flip_y=0.0001, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
        shuffle=True, random_state=None)

    M = computeM(c, alpha=0.5, beta=0.5, method='quasi_IPL')
    z = generateWeak(y, M, c)
    v = computeVirtual(z, c, method='quasi_IPL')

