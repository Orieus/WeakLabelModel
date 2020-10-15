from testUtils import openml_ids, load_dataset
from sklearn.linear_model import LogisticRegression


for problem, dataset_id in openml_ids.items():
    print('Evaluating {}[{}] dataset'.format(problem, dataset_id))

    X, y, n_classes, n_samples, n_features = load_dataset(problem)

    lr = LogisticRegression()
    lr.fit(X, y)
    print('Logistic Regression score = {}'.format(lr.score(X, y)))
