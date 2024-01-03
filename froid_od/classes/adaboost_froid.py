import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from froid_od.classes.froid import FROID


class adaboost_FROID(AdaBoostClassifier):
    def __init__(self, froid: FROID = FROID(), estimator=RandomForestClassifier(), *, n_estimators=50,
                 learning_rate=1.0, algorithm='SAMME.R', random_state=None, base_estimator='deprecated'):
        self.froid = froid
        self.estimator = estimator
        super().__init__(estimator=_adaboost_FROID_estimator(froid=froid, estimator=estimator),
                         n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm,
                         random_state=random_state, base_estimator=base_estimator)

class _adaboost_FROID_estimator(BaseEstimator, ClassifierMixin):
    def __init__(self, froid: FROID = FROID(), estimator=DecisionTreeClassifier()):
        self.froid = froid
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        self.y_ = y

        self.X_train_froid = np.hstack([self.froid.fit_transform(X), X])
        self.estimator.fit(self.X_train_froid, y, sample_weight=sample_weight)
        self.classes_ = self.estimator.classes_

        return self

    def predict(self, X):
        check_is_fitted(self)
        X_test_froid = np.hstack([self.froid.transform(X), X])

        return self.estimator.predict(X_test_froid)

    def predict_proba(self, X):
        check_is_fitted(self)
        X_test_froid = np.hstack([self.froid.transform(X), X])

        return self.estimator.predict_proba(X_test_froid)