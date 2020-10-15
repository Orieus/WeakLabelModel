# External modules
import os
import sys
import time
import errno

import scipy
import numpy as np
import sklearn.model_selection as skcv
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
import sklearn.datasets as skd
import openml

# Parameters for sklearn synthetic data
openml_ids = {'iris': 61, 'pendigits': 32, 'glass': 41, 'segment': 36,
              'vehicle': 54, 'vowel': 307, 'wine': 187, 'abalone': 1557,
              'balance-scale': 11, 'car': 21, 'ecoli': 39, 'satimage': 182,
              'collins': 478, 'cardiotocography': 1466, 'JapaneseVowels': 375,
              'autoUniv-au6-1000': 1555, 'autoUniv-au6-750': 1549,
              'analcatdata_dmft': 469, 'autoUniv-au7-1100': 1552,
              'GesturePhaseSegmentationProcessed': 4538,
              'autoUniv-au7-500': 1554, 'mfeat-zernike': 22, 'zoo': 62,
              'page-blocks': 30, 'yeast': 181, 'flags': 285,
              'visualizing_livestock': 685, 'diggle_table_a2': 694,
              'prnn_fglass': 952, 'confidence': 468, 'fl2000': 477}
openml_ids_nans = {'heart-c': 49, 'dermatology': 35}

def load_dataset(dataset, n_samples=1000, n_features=10, n_classes=2,
                 seed=None):
    if dataset in list(openml_ids.keys()):
        dataset_id = openml_ids[dataset]
        data = openml.datasets.get_dataset(dataset_id)
        X, y, categorical, feature_names = data.get_data(
                                target=data.default_target_attribute,
                                )
        # TODO change NaN in categories for another category
        categorical_indices = np.where(categorical)[0]
        ct = ColumnTransformer([("Name_Of_Your_Step",
                                 OneHotEncoder(),categorical_indices)],
                               remainder="passthrough") 
        X = ct.fit_transform(X)  # Categorical to binary
        n_samples = X.shape[0]           # Sample size
        n_features = X.shape[1]             # Data dimension
        # Assegurar que los valores en Y son correctos para todos los
        # resultados
        le = LabelEncoder()
        y = le.fit_transform(y)
        n_classes = y.max()+1      # Number of classes
    elif dataset == 'blobs':
        X, y = skd.make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=n_classes, cluster_std=1.0,
                              center_box=(-15.0, 15.0), shuffle=True,
                              random_state=seed)
    elif dataset == 'gauss_quantiles':
        X, y = skd.make_gaussian_quantiles(n_samples=n_samples,
                                           n_features=n_features,
                                           n_classes=n_classes,
                                           shuffle=True, random_state=seed)
    elif dataset == 'digits':
        X, y = skd.load_digits(n_class=n_classes, return_X_y=True)
        n_features = X.shape[0]             # Data dimension
    else:
        raise "Problem type unknown: {}"
    si = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = si.fit_transform(X)
    if type(X) is scipy.sparse.csc.csc_matrix:
        X = X.todense()
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    X, y = shuffle(X, y, random_state=seed)

    # ## Report data used in the simulation
    print('----------------')
    print('Dataset description:')
    print('    Dataset name: {0}'.format(dataset))
    print('    Sample size: {0}'.format(n_samples))
    print('    Number of features: {0}'.format(n_features))
    print('    Number of classes: {0}'.format(n_classes))

    return X, y, n_classes, n_samples, n_features


def newfig(name):
    fig = plt.figure(name, figsize=(3,3))
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


def plot_data(x, y, loc='best', save=True, title='data'):
    fig = newfig('data')
    ax = fig.add_subplot(111)
    ax.scatter(x[:, 0], x[:, 1], c=y, s=30, edgecolors=None, cmap='Paired',
               alpha=.8, lw=0.1)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_title(title)
    ax.set_xticks([-3,-2,-1,0,1,2,3])
    ax.set_yticks([-3,-2,-1,0,1,2,3])
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
    ax.set_xticks(list(range(1, 1 + len(tag_list))))
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

    y : array-like, with shape (n_samples, )
        The target variable of integers. This array is used for the evaluation
        of the model.

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
    for i in range(n_sim):
        # ##############
        # Self evaluation.
        # First, we compute leave-one-out predictions
        n_folds = min(10, ns)

        X_shuff, v_shuff, y_shuff = shuffle(X, v, y, random_state=i)
        # FIXME n_jobs can not be >1 because classif can not be serialized
        preds = skcv.cross_val_predict(classif, X_shuff, v_shuff, cv=n_folds,
                                       verbose=0, n_jobs=n_jobs)

        # Estimate error rates:s
        Pe_cv[i] = float(np.count_nonzero(y_shuff != preds)) / ns

        # ########################
        # Ground truth evaluation:
        #   Training with the given virtual labels (by default true labels)
        classif.fit(X, v)
        f = classif.predict_proba(X)

        # Then, we evaluate this classifier with all true labels
        # Note that training and test samples are being used in this error rate
        d = np.argmax(f, axis=1)
        Pe_tr[i] = float(np.count_nonzero(y != d)) / ns

        print(('\tAveraging {0} simulations. Estimated time to finish '
               '{1:0.4f}s.\r').format(n_sim,
                                      (time.clock() - start)/(i+1)*(n_sim-i)), end=' ')
        sys.stdout.flush()

    DEBUG = True
    if DEBUG:
        print(("y[:5] = {}".format(y_shuff[:5])))
        print(("q[:5] = {}".format(preds[:5])))
        print(("v[:5] = \n{}".format(v_shuff[:5])))
    print('')
    return Pe_tr, Pe_cv
