#!/usr/bin/env python
# -*- coding: utf-8 -*-

# External modules
import numpy as np
# from numpy import binary_repr
import sklearn.datasets as skd           # Needs version 0.14 or higher
from sklearn.preprocessing import label_binarize
# import sklearn.linear_model as sklm
import sys
# import ipdb
import cvxpy


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
                'quasi-IPL':    This is the quasi independent partial label
                                case discussed in the paper.

    Returns
    -------
    M : array-like, shape = (n_classes, n_classes)
    """

    if method == 'supervised':

        M = np.eye(c)

    elif method == 'noisy':

        M = (np.eye(c) * (alpha - (1 - alpha) / (c - 1))
             + np.ones((c, c)) * (1 - alpha) / (c - 1))

    elif method == 'random_noise':

        M = np.random.rand(c, c)
        M = M / np.sum(M, axis=0, keepdims=True)

        M = (1 - beta) * np.eye(c) + beta * M

    elif method == 'random_weak':

        # Number or rows. Equal to 2**c to simulate a scenario where all
        # possible binary label vectors are possible.
        d = 2**c

        # Supervised component: Identity matrix with size d x c.
        Ic = np.zeros((d, c))
        for i in range(c):
            Ic[2**(c - i - 1), i] = 1

        # Weak component: Random weak label proabilities
        M = np.random.rand(d, c)
        M = M / np.sum(M, axis=0, keepdims=True)

        # Averaging supervised and weak components
        M = (1 - beta) * Ic + beta * M

    elif method == 'IPL':

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(0, d):

            # Convert the decimal value z to a binary list of length c
            z_bin = np.array([int(b) for b in bin(z)[2:].zfill(c)])
            modz = sum(z_bin)

            M[z, :] = (alpha**(z_bin) * (1 - alpha)**(1 - z_bin)
                       * (beta**(modz - z_bin)
                       * (1 - beta)**(c - modz - 1 + z_bin)))

        # This is likely not required: columns in M should already sum up to 1
        M = M / np.sum(M, axis=0)

    elif method == 'IPL3':

        M = np.array([
            [0.0,                 0.0,               0.0],
            [0,                   0,                 (1 - gamma)**2],
            [0,                   (1 - beta)**2,     0],
            [0.0,                 beta * (1 - beta), gamma * (1 - gamma)],
            [(1 - alpha)**2,      0,                 0],
            [alpha * (1 - alpha), 0.0,               gamma * (1 - gamma)],
            [alpha * (1 - alpha), beta * (1 - beta), 0.0],
            [alpha**2,            beta**2,           gamma**2]])

    elif method == 'quasi-IPL':

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
        for z in range(1, d - 1):

            # Convert the decimal value z to a binary list of length c
            z_bin = [int(b) for b in bin(z)[2:].zfill(c)]
            modz = sum(z_bin)

            M[z, :] = z_bin * (beta**(modz - 1) * (1 - beta)**(c - modz))

        # Columns in M should sum up to 1
        M = M / np.sum(M, axis=0)

    else:
        raise ValueError("Unknown method to compute M: {}".format(method))

    return M


def generateM(c, method='supervised', alpha=0.2, beta=0.5):
    """
    Generate a mixing matrix M of a given type, given the number of classes c
    and some distribution parameters

    Parameters
    ----------
    c      : int
        Number of classes (i.e. number of columns in output matrix M)

    method : string, optional (default='supervised').
        Method to generate M. Available options are:
        - 'supervised': Identity matrix. For a fully labeled case.
        - 'noisy': For a noisy label case with deterministic parameters:
                The true label is observed with a given probability, otherwise
                one noisy label is taken at random. Parameter alpha is
                deterministic.
        - 'random_noise': Noisy labels with stochastic parameters.
                Same as 'noixy', but the parameters of the noise distribution
                are generated at random.
        - 'random_weak': A generic mixing label matrix with stochastic
                components
        - 'IPL': Independent partial labels: the observed labels are
                independent. The true label is observed with probability alpha.
                Each False label is observed with probability beta.
        - 'IPL3': It is a generalized version of IPL, but only for c=3 classes
                and alpha=1: each false label is observed with a different
                probability. Parameters alpha, beta and gamma represent the
                probability of a false label for each column.
        - 'quasi-IPL': This is the quasi-independent partial label case: the
                probability of any weak label depends on the number of false
                labels only.

    alpha : float in [0, 1] or array-like (size = c), optional (default=0.2)
        Noise degree parameter. Higher values of this parameter usually mean
        higher label noise.
        The specific meaning of this parameter depends on the method:
        - 'supervised': Ignored.
        - 'noisy': noise probability (i.e. probability that the weak label does
            not correspond to the true label).
            If array-like, this probability is class-dependent
        - 'random_noise': Noise probability (same as 'noisy')
        - 'random_weak': Weak label probability. It is the probability that the
            weak label is generated at random.
            If array-like, this probability is class-dependent.
        - 'IPL': Missing label probability. It is the probability that the true
            label is not observed in the weak label.
            If array-like, this probability is class-dependent.
        - 'IPL3': Ignored
        - 'quasi-IPL': Ignored.

    beta : float (non-negative) or array-like, optional (default=0.5)
        Noise distribution parameter.
        The specific meaning of this parameter depends on the method:
        - 'supervised': Ignored.
        - 'noisy': Ignored
        - 'random_noise': Concentration parameter. The noisy label
            probabilities are generated stochastically according to a Dirichlet
            distribution with parameters beta. According to this:
                - beta = 1 is equivalent to a uniform distribution
                - beta = inf is equivalent to using option 'noisy': the class
                    of the noisy label is random.
                - beta < 1 implies higher concentration: most noise probability
                    gets concentrated in a single class. This may be usefult to
                    simulate situations where a class is usually mixed with
                    another similar clas, but not with others.
            If beta is array-like, a different concentration parameter will be
            used for each class (i.e. for each column of M)
        - 'random_weak': Concentration parameter of the weak label probability
            distribution, which is a Dirichlet.
                - beta = 1 is equivalent to a uniform distribution
                - beta = inf is equivalent to a constant probability over all
                    weak labels
                - beta < 1 implies higher concentration: most probability mass
                    is concentrated over a few weak labels
            If beta is array-like, a different concentration parameter will be
            used for each class (i.e. for each column of M)
        - 'IPL': Probability that a noisy label from a given class is observed
            If array-like, this probability is class-dependent: beta[c] is the
            probability that, if the true label is not c, the weak label
            contains c
        - 'IPL3': Probability that a noisy label from any class is observed.
            If array-like, this probability is class-dependent: beta[c] is the
            probability that, if the true label is c, the weak label
            contains a label from class c' other than c
        - 'quasi-IPL': Ignored.

    Returns
    -------
    M : array-like, shape = (n_classes, n_classes)
    """

    if method == 'supervised':

        M = np.eye(c)

    elif method == 'noisy':

        valpha = np.array(alpha)
        M = (np.eye(c) * (1 - valpha - valpha / (c - 1))
             + np.ones((c, c)) * valpha / (c - 1))

    elif method == 'random_noise':

        # Diagonal component (no-noise probabilities)
        # np.array is used just in case beta is a list
        D = (1 - np.array(alpha)) * np.eye(c)

        # Non-diagonal components
        # Transforma beta into an np.array (if it isn't it).
        vbeta = np.array(beta) * np.ones(c)
        B = np.random.dirichlet(vbeta, c).T

        # Remove diagonal component and rescale
        # I am using here the fact that the conditional distribution of a
        # rescaled subvector of a dirichlet is a dirichet with the same
        # parameters, see
        # https://math.stackexchange.com/questions/1976544/conditional-
        # distribution-of-subvector-of-a-dirichlet-random-variable
        # Conditioning...
        B = B * (1 - np.eye(c))
        # Rescaling...
        B = B / np.sum(B, axis=0)
        # Rescale by (1-beta), which are the probs of noisy labels
        B = B @ (np.eye(c) - D)

        # Compute M
        M = D + B

    elif method == 'random_weak':

        # Number or rows. Equal to 2**c to simulate a scenario where all
        # possible binary label vectors are possible.
        d = 2**c

        # Supervised component: Identity matrix with size d x c.
        Ic = np.zeros((d, c))
        for i in range(c):
            Ic[2**(c - i - 1), i] = 1

        # Weak component: Random weak label proabilities
        # Transforma beta into an np.array (if it isn't it).
        vbeta = np.array(beta) * np.ones(d)
        B = np.random.dirichlet(vbeta, c).T

        # Averaging supervised and weak components
        # np.array is used just in case alpha is a list
        M = (1 - np.array(alpha)) * Ic + np.array(alpha) * B

    elif method == 'IPL':

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        valpha = np.array(alpha)
        vbeta = np.array(beta)

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(0, d):

            # Convert the decimal value z to a binary list of length c
            z_bin = np.array([int(b) for b in bin(z)[2:].zfill(c)])
            modz = sum(z_bin)

            M[z, :] = (((1 - valpha) / vbeta)**z_bin
                       * (valpha / (1 - vbeta))**(1 - z_bin)
                       * np.prod(vbeta**z_bin)
                       * np.prod((1 - vbeta)**(1 - z_bin)))

    elif method == 'IPL3':

        b0 = beta[0]
        b1 = beta[1]
        b2 = beta[2]

        M = np.array([
            [0.0, 0.0, 0.0],
            [0, 0, (1 - b2)**2],
            [0, (1 - b1)**2, 0],
            [0.0, b1 * (1 - b1), b2 * (1 - b2)],
            [(1 - b0)**2, 0, 0],
            [b0 * (1 - b0), 0.0, b2 * (1 - b2)],
            [b0 * (1 - b0), b1 * (1 - b1), 0.0],
            [b0**2, b1**2, b2**2]])

    elif method == 'quasi-IPL':

        beta = np.array(beta)

        # Shape M
        d = 2**c
        M = np.zeros((d, c))

        # Compute mixing matrix row by row for the nonzero rows
        for z in range(1, d - 1):

            # Convert the decimal value z to a binary list of length c
            z_bin = np.array([int(b) for b in bin(z)[2:].zfill(c)])
            modz = sum(z_bin)

            M[z, :] = z_bin * (beta**(modz - 1) * (1 - beta)**(c - modz))

        # Columns in M should sum up to 1
        M = M / np.sum(M, axis=0)

    else:
        raise ValueError(f"Unknown method to compute M: {method}")

    return M


def generateWeak(y, M, dec_labels=None):
    """
    Generate the set of weak labels z from the ground truth labels y, given
    a mixing matrix M and, optionally, a set of possible weak labels, zset.

    Args:
        y       :List of true labels with values from 0 to c-1 where c is the
                 number of classes
        M       :Mixing matrix of shape (d, c) with d >= c.
        dec_labels :A list of indices in {0, 1, ..., 2**c}: dec_labels[i] is an
                 integer whose binary representation encodes the weak labels
                 corresponding to the i-th row in M. The length of dec_labels
                 must be equal to the number of rows in M.

                 If dec_labels is None: the following is assumed:
                   - If M is (2**c, c), dec_labels = [0, 1, ..., 2**c]
                   - If M is (c, c),    dec_labels = [1, 2, 4,..., 2**(c-1)]
                   - Otherwise, a error is raised.

    Returns:
        z   :List of weak labels. Each weak label is an integer whose binary
            representation encodes the observed weak labels.
    """

    z = np.zeros(y.shape, dtype=int)  # Weak labels for all labels y (int)
    d = M.shape[0]               # Number of weak labels
    c = M.shape[1]

    if dec_labels is None:
        if d == 2**c:
            dec_labels = np.arange(2**c)
        elif d == c:
            dec_labels = 2**np.arange(c-1, -1, -1)
        else:
            raise ValueError(
                "A dec_labels parameter is required for the given M")

    # dec_labels = np.arange(d)    # Possible weak labels (int)
    for index, i in enumerate(y):
        z[index] = np.random.choice(dec_labels, 1, p=M[:, i])

    # if c == d:
    #     z = 2**(c-z-1)

    return z


def binarizeWeakLabels(z, c):
    """
    Binarizes the weak labels depending on the method used to generate the weak
    labels.

    Args:
        z       :List of weak labels. Each weak label is an integer whose
                 binary representation encondes the observed weak labels
        c       :Number of classes. All components of z must be smaller than
                 2**c
    Returns:
        z_bin
    """
    # Transform the weak label indices in z into binary label vectors
    z_bin = np.zeros((z.size, c), dtype=int)       # weak labels (binary)
    for index, i in enumerate(z):         # From dec to bin
        z_bin[index, :] = [int(x) for x in np.binary_repr(i, width=c)]

    return z_bin


def computeVirtualMatrixOptimized(weak_labels, mixing_matrix, convex=True):
    """
    Parameters
    ----------
    weak_labels : (n_samples, n_weak_labels) numpy.ndarray
        Binary indication matrix with only one one per row indicating
        to which class the instance belongs to.
    mixing_matrix : (n_weak_labels, n_true_labels) numpy.ndarray
        Mixing matrix of floats corresponding to the stochastic
        process that generates the weak labels from the true labels.
    Convex : boolean

    Returns
    -------
    virtual_matrix : (n_samples, n_weak_labels) numpy.ndarray
    """
    d, c = mixing_matrix.shape
    p = np.sum(weak_labels, 0) / np.sum(weak_labels)
    I = np.eye(c)
    c1 = np.ones([c, 1])
    d1 = np.ones([d, 1])
    if convex is True:
        hat_Y = cvxpy.Variable((c, d))
        prob = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.norm(cvxpy.hstack(
                [cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]), 1)),
            [hat_Y @ mixing_matrix == I, hat_Y.T @ c1 == d1])
    else:
        hat_Y = cvxpy.Variable((c, d))
        prob = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.norm(cvxpy.hstack(
                [cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]), 1)),
            [hat_Y @ mixing_matrix == I])
    prob.solve()
    return hat_Y.value


# FIXME change reference to Optimized to a useful name
def computeVirtual(z, c, method='IPL', M=None, dec_labels=None):
    """
    Generate the set of virtual labels v for the (decimal) weak labels in z,
    given a weak label model in variable method and, optionally, a mixing
    matrix M, and a list of admissible decimal labels.

    Args:
        z       :List of weak labels. Each weak label is an integer whose
                 binary representation encondes the observed weak labels
        c       :Number of classes. All components of z must be smaller than
                 2**c
        method  :Method applied to compute the virtual label vector v.
                 Available methods are:
                 - 'supervised' :Takes virtual label vectors equal to the
                                 binary representations of the weak labels in z

                 - 'IPL'        :Independet Partial Labels. Equivalent to
                                 supervised
                 - 'quasi-IPL'  :Computes virtual labels assuming that the
                                 mixing matrix M was 'quasi-IPL' without
                                 knowing the M
                 - 'known-M-pseudo'    :Computes virtual labels for a M-proper loss.
                 - 'MCC'        :Computes virtual labels for a M-CC loss
                                 (Not available yet)
                 - 'known-M-opt'       :Computes virtual labels with the opt method
                 - 'known-M-opt-conv'  :Computes virtual labels with the opt method and convex
        M       :Mixing matrix. Only for methods 'Mproper' and 'MCC'
        dec_labels :A list of indices in {0, 1, ..., 2**c}: dec_labels[i] is an
                 integer whose binary representation encodes the weak labels
                 corresponding to the i-th row in M. The length of dec_labels
                 must be equal to the number of rows in M.

                 If dec_labels is None: the following is assumed:
                   - If M is (2**c, c), dec_labels = [0, 1, ..., 2**c]
                   - If M is (c, c),    dec_labels = [1, 2, 4,..., 2**(c-1)]
                   - Otherwise, a error is raised.

    Returns:
        v
    """

    if method in ['supervised', 'IPL']:
        v = binarizeWeakLabels(z, c).astype(float)
    elif method == 'quasi-IPL':    # quasi-independent labels

        # The virtual labels are computed from the weak label vectors
        v = binarizeWeakLabels(z, c).astype(float)

        # Each 1 or 0 in the weak label vector must be replaced by a number
        # that depends on the total number of 1's in the vector
        for index in range(len(v)):
            aux = v[index, :]
            weak_sum = np.sum(aux)
            if weak_sum != c:
                weak_zero = float(1 - weak_sum)/(c - weak_sum)
                aux[aux == 0] = weak_zero
                v[index, :] = aux
            else:
                # In the quasi-IPL method, it is assumed that nor z=0 nor
                # z=2**C will happen. A zero vector is assigned here, just in
                # case, though the choice is arbitrary.
                # TODO MPN I changed Nans to zeros. Is this important?
                v[index, :] = np.array([None] * c)

    elif method in ['known-M-pseudo', 'known-M-opt', 'known-M-opt-conv']:
        # Compute array of all possible weak label vectors (in decimal format)
        # in the appropriate order, if not given.
        if dec_labels is None:
            if M.shape[0] == 2**c:
                # All possible weak labels have a row in M
                dec_labels = np.arange(2**c)
            elif M.shape[0] == c:
                # Single-class label vectors are assumed
                dec_labels = 2**np.arange(c - 1, -1, -1)
            else:
                raise ValueError("Weak labels for the given M are unknown")

        # Compute inverted index from decimal labels to position in dec_labels
        z2i = dict(list(zip(dec_labels, list(range(len(dec_labels))))))

        # Compute the virtual label matrix
        if method == 'known-M-pseudo':
            Y = np.linalg.pinv(M)
        elif method == 'known-M-opt':
            binary_z = label_binarize(z, range(2**c))
            Y = computeVirtualMatrixOptimized(binary_z, M, convex=False)
        elif method == 'known-M-opt-conv':
            binary_z = label_binarize(z, range(2**c))
            Y = computeVirtualMatrixOptimized(binary_z, M, convex=True)

        # THIS IS NO LONGER REQUIRD
        # If mixing matrix is square, weak labels need to be transformed from
        # 2**c to c optional values
        # if M.shape[0] == M.shape[1]:
        #     z = c-np.log2(z)-1

        # Compute the virtual label.
        v = np.zeros((z.size, c))
        for i, zi in enumerate(z):
            # The virtual label for the i-th weak label, zi, is the column
            # in Y corresponding to zi (that is taken from the inverted index)
            v[i, :] = Y[:, z2i[zi]]
    else:
        raise ValueError(
            "Unknown method to create virtual labels: {}".format(method))

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

    print("=======================")
    print("    Weak labels")
    print("=======================")

    ###########################################################################
    # ## PART I: Load data (samples and true labels)                         ##
    ###########################################################################

    X, y = skd.make_classification(
        n_samples=ns, n_features=nf, n_informative=2, n_redundant=0,
        n_repeated=0, n_classes=c, n_clusters_per_class=1, weights=None,
        flip_y=0.0001, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
        shuffle=True, random_state=None)

    M = computeM(c, alpha=0.5, beta=0.5, method='quasi-IPL')
    z = generateWeak(y, M, c)
    v = computeVirtual(z, c, method='quasi-IPL')

    print(M)
    print(z)
    print(v)

    ipdb.set_trace()


if __name__ == "__main__":

    main()

