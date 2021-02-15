# External modules
import sys
import time

import numpy as np
import sklearn.model_selection as skcv
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


def evaluateClassif(classif, X, y,  v=None, n_sim=1, n_jobs=1, n_folds = 5):
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
    Pe_test = [0] * n_sim * n_folds
    #Pe_cv = [0] * n_sim * n_folds
    historia = []

    ns = X.shape[0]
    #test_ns = X_test.shape[0]
    start = time.time()
    # ## Loop over simulation runs
    for i in range(n_sim):
        # ##############
        # Self evaluation.
        # First, we compute leave-one-out predictions
        n_folds = min(10, ns)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=i)
        kf.get_n_splits(X)
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            v_train, v_test = v[train_index], v[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # ########################
            # Ground truth evaluation:
            #   Training with the given virtual labels (by default true labels)
            hist = classif.fit(X_train, v_train)
            probas = classif.predict_proba(X_test)

            # Then, we evaluate this classifier with all true labels
            # Note that training and test samples are being used in this error rate
            predictions = np.argmax(probas, axis=1)
            Pe_test[i] = np.mean(y_test != predictions)
            historia.append(hist)

    print(('\tAveraging {0} simulations. Estimated time to finish '
               '{1:0.4f}s.\r').format(n_sim,
                                      (time.time() - start)/(i+1)*(n_sim-i)),
              end=' ')
    sys.stdout.flush()

    DEBUG = True
    if DEBUG:
        print(("y[:5] = {}".format(y_test[:5])))
        print(("q[:5] = {}".format(predictions[:5])))
        print(("v[:5] = \n{}".format(v_test[:5])))
    print('')
    return Pe_test, historia
