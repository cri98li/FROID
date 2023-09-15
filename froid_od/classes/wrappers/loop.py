import numpy as np
from PyNomaly.loop import LocalOutlierProbability
from tqdm.auto import tqdm


class FROID_LocalOutlierProbability:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X_train:np.ndarray):
        self.X_train = X_train
        self.loop = LocalOutlierProbability(self.X_train, **self.kwargs).fit()

        return self

    def transform(self, X_test):
        res = []

        for el in tqdm(X_test, disable=True):
            res.append(self.loop.stream(el))

        return np.array(res)

    def transform_one_by_one(self, X_test):
        X = np.vstack([self.X_train, X_test])
        idxs = [i for i in range(len(self.X_train))]

        res = []

        for i in tqdm(range(len(X_test)), desc="LoOP transform"):
            loop = LocalOutlierProbability(X[idxs+[i+len(self.X_train)]], **self.kwargs).fit()
            res.append(loop.local_outlier_probabilities[len(self.X_train)])

        return np.array(res).T

    def transform_all(self, X_test):
        self.loop = LocalOutlierProbability(np.vstack([self.X_train, X_test]), **self.kwargs).fit()
        return self.loop.local_outlier_probabilities[len(self.X_train):]

    def decision_function(self, X):
        return self.transform(X)