# External modules
import sys
import time

import numpy as np
import sklearn.model_selection as skcv
from sklearn.utils import shuffle


def evaluateClassif(classif, X, y, X_test, y_test,  v=None, n_sim=1, n_jobs=1):
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
    Pe_test = [0] * n_sim
    Pe_cv = [0] * n_sim
    historia = []

    ns = X.shape[0]
    test_ns = X_test.shape[0]
    start = time.time()
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
        hist = classif.fit(X, v)
        f = classif.predict_proba(X_test)

        # Then, we evaluate this classifier with all true labels
        # Note that training and test samples are being used in this error rate
        d = np.argmax(f, axis=1)
        Pe_test[i] = float(np.count_nonzero(y_test != d)) / test_ns
        historia.append(hist)

        print(('\tAveraging {0} simulations. Estimated time to finish '
               '{1:0.4f}s.\r').format(n_sim,
                                      (time.time() - start)/(i+1)*(n_sim-i)),
              end=' ')
        sys.stdout.flush()

    DEBUG = True
    if DEBUG:
        print(("y[:5] = {}".format(y_shuff[:5])))
        print(("q[:5] = {}".format(preds[:5])))
        print(("v[:5] = \n{}".format(v_shuff[:5])))
    print('')
    return Pe_test, Pe_cv, historia
