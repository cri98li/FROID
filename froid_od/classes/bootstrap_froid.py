import random

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.base import clone
from tqdm.auto import tqdm

from froid_od.classes.froid import FROID
from numpy.random import choice


class bootstrap_FROID(BaseEstimator, TransformerMixin):
    def __init__(self, n_resamples=10, resample_rows_size=.2, resample_cols_size=.2, replace=True, p=None,
                 froid_base: FROID = FROID(), random_fr=1., random_od=1., verbose=False, random_state=42):
        self.random_state = random_state
        self.n_resamples = n_resamples
        self.resample_rows_size = resample_rows_size
        self.resample_cols_size = resample_cols_size
        self.froid_base = froid_base
        self.random_fr = random_fr
        self.random_od = random_od
        self.replace = replace
        self.p = p
        self.verbose = verbose

    def fit(self, X):
        random.seed(self.random_state)

        self.froid_instances = []

        n_fr = int(min(len(self.froid_base.feat_red) / self.random_fr, 1)) if type(
            self.random_fr) is float else self.random_fr
        n_od = int(min(len(self.froid_base.out_det) / self.random_od, 1)) if type(
            self.random_od) is float else self.random_od

        random.seed(self.random_state)
        for _ in range(self.n_resamples):
            froid_clone = clone(self.froid_base)

            froid_clone.feat_red = dict(random.sample(sorted(froid_clone.feat_red.items()), n_fr))
            froid_clone.out_det = dict(random.sample(sorted(froid_clone.out_det.items()), n_od))

            self.froid_instances.append(froid_clone)

        self.froid_instances = [clone(self.froid_base) for _ in range(self.n_resamples)]
        n_rows = list(range(len(X)))
        n_cols = list(range(len(X[0])))

        if type(self.resample_rows_size) != int:
            self.resample_rows_size = max(int(self.resample_rows_size * len(n_rows)), 2)
        if type(self.resample_cols_size) != int:
            self.resample_cols_size = max(int(self.resample_cols_size * len(n_cols)), 2)

        self.rows_idxs = [choice(n_rows, self.resample_rows_size, self.replace, self.p) for _ in
                          range(self.n_resamples)]
        self.cols_idxs = [choice(n_cols, self.resample_cols_size, False) for _ in
                          range(self.n_resamples)]

        for row_idx, col_idx, froid_instance in zip(
                tqdm(self.rows_idxs, disable=not self.verbose, desc="fitting bootstrap"),
                self.cols_idxs,
                self.froid_instances):
            froid_instance.fit(X[row_idx][:, col_idx])

        return self


    def transform(self, X):
        to_append_horizontally = []

        for i, (froid_instance, col_idx) in enumerate(zip(
                tqdm(self.froid_instances, disable=not self.verbose, desc="transform bootstrap"), self.cols_idxs)):
            res = froid_instance.transform(X[:, col_idx])
            to_append_horizontally.append(res)

        return np.hstack(to_append_horizontally)
