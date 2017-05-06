# External modules
import os
import sys
import time
import errno

import numpy as np
import sklearn.cross_validation as skcv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def newfig(name):
    fig = plt.figure(name)
    fig.clf()
    return fig


def savefig(fig, path='figures', prefix='weak_labels_', extension='svg'):
    fig.tight_layout()
    name = fig.get_label()
    filename = "{}{}.{}".format(prefix, name, extension)
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    fig.savefig(os.path.join(path, filename))


def plot_data(x, y, loc='best', save=True):
    fig = newfig('data')
    ax = fig.add_subplot(111)
    ax.scatter(x[:, 0], x[:, 1], c=y, s=50, edgecolors='black', cmap='Paired',
               alpha=.8)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_title('Labeled dataset')
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc=loc)
    if save:
        savefig(fig)
    return fig


def get_grid(X, delta=1.0):
    x1_min, x2_min = X.min(axis=0)
    x1_max, x2_max = X.max(axis=0)
    x1_grid = np.arange(x1_min, x1_max, delta)
    x2_grid = np.arange(x2_min, x2_max, delta)
    MX1, MX2 = np.meshgrid(x1_grid, x2_grid)
    x_grid = np.asarray([MX1.flatten(), MX2.flatten()]).T.reshape(-1, 2)
    return x_grid, MX1, MX2


def ax_scatter_question(ax, x, notes, zorder=2, clip_on=True, text_size=18,
                        color='w'):
    ax.scatter(x[:, 0], x[:, 1], c=color, s=500, zorder=zorder,
               clip_on=clip_on)
    for point, n in zip(x, notes):
        ax.annotate('{}'.format(n), xy=(point[0], point[1]),
                    xytext=(point[0], point[1]), ha="center", va="center",
                    size=text_size, zorder=zorder, clip_on=clip_on)


def plot_data_predictions(fig, x, y, Z, MX1, MX2, x_predict=None, notes=None,
                          cmap=cm.YlGnBu, cmap_r=cm.plasma,
                          loc='best', title=None, aspect='auto', s=30):
    x1_min, x1_max = MX1.min(), MX1.max()
    x2_min, x2_max = MX2.min(), MX2.max()

    ax = fig.add_subplot(111)
    ax.scatter(x[y == 0, 0], x[y == 0, 1], c='y', s=s, alpha=0.5,
               label='Train. Class 1')
    ax.scatter(x[y == 1, 0], x[y == 1, 1], c='b', s=s, alpha=0.5,
               label='Train. Class 2')
    ax.grid(True)

    # Colormap
    ax.imshow(Z, interpolation='bilinear', origin='lower', cmap=cmap_r,
              extent=(x1_min, x1_max, x2_min, x2_max), alpha=0.3,
              aspect=aspect)

    # Contour
    # FIXME make levels optional
    # CS = ax.contour(MX1, MX2, Z, levels=[0.01, 0.25, 0.5, 0.75, 0.99],
    #                cmap=cmap)
    CS = ax.contour(MX1, MX2, Z, cmap=cmap)
    ax.clabel(CS, fontsize=13, inline=1)
    ax.set_xlim([x1_min, x1_max])
    ax.set_ylim([x2_min, x2_max])

    if x_predict is not None:
        ax_scatter_question(ax, x_predict, notes)
    if title is not None:
        ax.set_title(title)
    ax.legend(loc=loc)


def plot_results(tag_list, Pe_tr, Pe_cv, ns, n_classes, n_sim, loc='best',
        save=True):
    # Config plots.
    font = {'family': 'Verdana', 'weight': 'regular', 'size': 10}
    matplotlib.rc('font', **font)

    # Plot error scatter.
    fig = newfig('error_rate')
    ax = fig.add_subplot(111)
    for i, tag in enumerate(tag_list):
        ax.scatter([i + 1]*n_sim, Pe_tr[tag], c='white', edgecolors='black',
                   s=100, alpha=.8, label='training')
        ax.scatter([i + 1]*n_sim, Pe_cv[tag], c='black', edgecolors='black',
                   s=30, alpha=.8, label='validation')

    ax.set_title('Error rate, samples={}, classes={}, iterations={}'.format(ns,
                 n_classes, n_sim))
    ax.set_xticks(range(1, 1 + len(tag_list)))
    ax.set_xticklabels(tag_list, rotation=45, ha='right')
    ax.set_ylim([-0.01, 1.01])
    ax.legend(['training', 'validation'], loc=loc)
    ax.grid(True)
    if save:
        savefig(fig)
    return fig


def evaluateClassif(classif, X, y, v=None, n_sim=1, n_jobs=1):
    """Evaluates a classifier using cross-validation

    Parameters
    ----------
    classif : object
        This is the classifier that needs to be trained and evaluated. It needs
        to have the following functions:
            - fit(X,y) :
            - predict(X) :
            - predict_proba(X) :
            - get_params() : All the necessary parameters to create a deep copy

    X : array-like, with shape (n_samples, n_dim)
        The data to fit.

    y : array-like, with shape (n_samples, n_classes)
        The target variable. This array is used for the evaluation of the
        model.

    v : array-like, optional, with shape (n_samples, n_classes), default: 'y'
        The virtual target variable. This array is used for the training of the
        model.

    n_sim : integer, optional, default: 1
        The number of simulation runs.

    n_jobs : integer, optional, default: 1
        The number of CPUs to use to do the computation. -1 means 'all CPUs'

    Returns
    -------
    predictions_training : ndarray
        This are the predictions on the training set after training

    predictions_validation : ndarray
        This are the predictions on the validation set after training
    """
    # Default v
    if v is None:
        v = y

    # ## Initialize aggregate results
    Pe_tr = [0] * n_sim
    Pe_cv = [0] * n_sim

    ns = X.shape[0]
    start = time.clock()
    # ## Loop over simulation runs
    for i in xrange(n_sim):
        # ########################
        # Ground truth evaluation:
        #   Training with the given virtual labels (by default true labels)
        classif.fit(X, v)
        f = classif.predict_proba(X)

        # Then, we evaluate this classifier with all true labels
        # Note that training and test samples are being used in this error rate
        d = np.argmax(f, axis=1)
        Pe_tr[i] = float(np.count_nonzero(y != d)) / ns

        # ##############
        # Self evaluation.
        # First, we compute leave-one-out predictions
        n_folds = min(10, ns)
        preds = skcv.cross_val_predict(classif, X, v, cv=n_folds, verbose=0,
                                       n_jobs=n_jobs)

        # Estimate error rates:s
        Pe_cv[i] = float(np.count_nonzero(y != preds)) / ns

        print ('\tAveraging {0} simulations. Estimated time to finish '
               '{1:0.4f}s.\r').format(n_sim,
                                      (time.clock() - start)/(i+1)*(n_sim-i)),
        sys.stdout.flush()

    print ''
    return Pe_tr, Pe_cv
