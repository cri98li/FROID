import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble._bagging import BaseBagging
from sklearn.utils.validation import check_is_fitted

from froid_od.classes.froid import FROID


class bagging_FROID(BaggingClassifier):
    def __init__(self, froid: FROID = FROID(), estimator=RandomForestClassifier(), n_estimators=10, *, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                 n_jobs=None, random_state=None, verbose=False, base_estimator='deprecated'):
        self.froid = froid
        self.estimator = estimator
        super().__init__(estimator=_FROID_ensemble_estimator(froid=froid, estimator=estimator),
                         n_estimators=n_estimators, max_samples=max_samples,
                         max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features,
                         oob_score=oob_score, warm_start=warm_start, n_jobs=n_jobs, random_state=random_state,
                         verbose=verbose, base_estimator=base_estimator)

class _FROID_ensemble_estimator(BaseEstimator, ClassifierMixin):
    def __init__(self, froid: FROID = FROID(), estimator=RandomForestClassifier()):
        self.froid = froid
        self.estimator = estimator

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y

        self.X_train_froid = np.hstack([self.froid.fit_transform(X), X])
        self.estimator.fit(self.X_train_froid, y)

        return self


    def predict(self, X):
        check_is_fitted(self)
        X_test_froid = np.hstack([self.froid.transform(X), X])

        return self.estimator.predict(X_test_froid)