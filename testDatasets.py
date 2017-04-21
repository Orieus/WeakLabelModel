import openml

import sklearn.datasets as skd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

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


for problem, dataset_id in openml_ids.items():
    print('Evaluating {}[{}] dataset'.format(problem, dataset_id))
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical = dataset.get_data(
                                    target=dataset.default_target_attribute,
                                    return_categorical_indicator=True)
    # TODO change NaN in categories for another category
    enc = OneHotEncoder(categorical_features=categorical, sparse=False)
    try:
        X = enc.fit_transform(X)  # Categorical to binary
    except ValueError as ve:
        print ve
        from IPython import embed; embed()
    ns = X.shape[0]           # Sample size
    nf = X.shape[1]             # Data dimension
    n_classes = y.max()+1      # Number of classes
    n_it = 20            # Number of iterations

    X = Imputer(missing_values='NaN', strategy='mean').fit_transform(X)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    lr = LogisticRegression()
    lr.fit(X, y)
    print('Logistic Regression score = {}'.format(lr.score(X, y)))
