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

    Parameters
    ----------
    c      : int. Number of classes
    alpha  : float, optional (default=0.5)
    beta   : float, optional (default=0.5)
    gamma  : float, optional (default=0.5)
    method : string, optional (default='supervised'). Method to compute M.
             Available options are:
                'supervised':   Identity matrix. For a fully labeled case.
                'noisy':        For a noisy label case: the true label is
                                observed with probabiltity 1 - beta, otherwise
                                one noisy label is taken at random.
                'random_noise': All values of the mixing matrix are taken at
                                random from a uniform distribution. The matrix
                                is normalized to be left-stochastic
                'IPL':          Independent partial labels: the observed labels
                                are independent. The true label is observed
                                with probability alfa. Each False label is
                                observed with probability beta.
                'IPL3':         It is a generalized version of IPL, but only
                                for c=3 classes and alpha=1: each false label
                                is observed with a different probability.
                                Parameters alpha, beta and gamma represent the
                                probability of a false label for each column.
                'quasi_IPL':    This is the quasi independent partial label
                                case discussed in the paper.

    Returns
    -------
    M : array-like, shape = (n_classes, n_classes)
    """

    if method == 'supervised':

        M = np.eye(c)

    elif method == 'noisy':

        M = (np.eye(c) * (1-beta-beta/(c-1)) +
             np.ones((c, c)) * beta/(c-1))

    elif method == 'random_noise':

        # FIXME I thought the rows should sum to one... See the reason
        M = np.random.rand(c, c)
        M = M / np.sum(M, axis=0, keepdims=True)

    elif method == 'IPL':

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(0, d):

            # Convert the decimal value z to a binary list of length c
            z_bin = np.array([int(b) for b in bin(z)[2:].zfill(c)])
            modz = sum(z_bin)

            M[z, :] = (alpha**(z_bin) * (1-alpha)**(1-z_bin) *
                       (beta**(modz-z_bin) * (1-beta)**(c-modz-1+z_bin)))

        # This is likely not required: columns in M should already sum up to 1
        M = M / np.sum(M, axis=0)

    elif method == 'IPL3':

        M = np.array([
                [0.0,             0.0,           0.0],
                [0,               0,             (1-gamma)**2],
                [0,               (1-beta)**2,   0],
                [0.0,             beta*(1-beta), gamma*(1-gamma)],
                [(1-alpha)**2,    0,             0],
                [alpha*(1-alpha), 0.0,           gamma*(1-gamma)],
                [alpha*(1-alpha), beta*(1-beta), 0.0],
                [alpha**2,        beta**2,       gamma**2]])

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

            # Convert the decimal value z to a binary list of length c
            z_bin = [int(b) for b in bin(z)[2:].zfill(c)]
            modz = sum(z_bin)

            M[z, :] = z_bin*(beta**(modz-1) * (1-beta)**(c-modz))

        # This is likely not required: columns in M should already sum up to 1
        M = M / np.sum(M, axis=0)

    else:

        print "Error: unknown value of input argumen method in computeM"
        sys.exit()

    return M


def generateWeak(y, M, c):
    """
    Generate the set of weak labels z for n examples, given the ground truth
    labels y for n examples, a mixing matrix M, and a number of classes c.
    """

    z = np.zeros(y.shape, dtype=int)  # Weak labels for all labels y (int)
    d = M.shape[0]               # Number of weak labels
    # d = 2 ** c                   # Number of weak labels
    dec_labels = np.arange(d)    # Possible weak labels (int)

    for index, i in enumerate(y):
        z[index] = np.random.choice(dec_labels, 1, p=M[:, i])

    return z

def dec_to_bin(z, c):
    return computeVirtual(z, c, method='IPL', M=None)

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

    z_bin = np.zeros((z.size, c), dtype=int)         # weak labels (binary)
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

