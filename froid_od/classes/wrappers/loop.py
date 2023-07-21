import numpy as np
from PyNomaly.loop import LocalOutlierProbability


class FROID_LocalOutlierProbability:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X_train:np.ndarray):
        self.X_train = X_train

        return self

    def transform(self, X_test):
        self.loop = LocalOutlierProbability(np.vstack([self.X_train, X_test]), **self.kwargs).fit()
        return self.loop.local_outlier_probabilities[len(self.X_train):]

    def decision_function(self, X):
        return self.transform(X)