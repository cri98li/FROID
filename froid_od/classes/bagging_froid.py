from sklearn.base import TransformerMixin, BaseEstimator


class FROID_ensemble(BaseEstimator, TransformerMixin):
    def __init__(self, froid_iter=1, seed=42, n_jobs=1, **kwargs):
        pass