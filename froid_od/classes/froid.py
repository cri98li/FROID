import copy

import numpy as np
from tqdm.auto import tqdm

from pyod.models.cblof import CBLOF
from pyod.models.pca import PCA as pyod_PCA
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.suod import SUOD
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS, TSNE, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.neighbors import LocalOutlierFactor
from sklearn.random_projection import GaussianRandomProjection

from froid_od.classes.wrappers.loop import FROID_LocalOutlierProbability


class FROID(BaseEstimator, TransformerMixin):

    out_det_dict = {
        "knn": KNN,
        "lof": LocalOutlierFactor,
        "loop": FROID_LocalOutlierProbability,
        "cof": COF,
        #"cblof": CBLOF,  # Il clustering non converge
        "ellenv": EllipticEnvelope,
        "hbos": HBOS,
        "mcd": MCD,
        "featurebagging": FeatureBagging,
        "copod": COPOD,
        "isofor": IForest,
        "loda": LODA,
        #"suod": SUOD, # problemi di versioni
        "ocsvm": OCSVM,
        "pca": pyod_PCA,
    }

    feat_red_dict = {
        "pca": PCA,
        "isomap": Isomap,
        #"mds": MDS, #aggiungere il metodo transforms
        "kpca": KernelPCA,
        #"tsne": TSNE, #aggiungere il metodo transforms
        #"rsp": GaussianRandomProjection, # vedere gli iperparametri
        "lle": LocallyLinearEmbedding,
        #"se": SpectralEmbedding, #aggiungere il metodo transforms
    }
    def __init__(self, out_det="all", feat_red="all", froid_iter=1, seed=42, n_jobs=1, **kwargs):
        self.n_jobs = n_jobs
        self.seed = seed
        self.kwargs = kwargs
        self.iter = froid_iter

        self.fitted_ods = []
        self.fitted_frs = []

        self.kwargs["n_jobs"] = n_jobs
        self.kwargs["seed"] = seed
        self.kwargs["random_state"] = seed

        if "novelty" not in self.kwargs:
            self.kwargs["novelty"] = True

        if type(out_det) == str and out_det == "all":
            self.out_det = FROID.out_det_dict
        elif type(out_det) == dict:
            self.out_det = out_det
        elif type(out_det) == list:
            self.out_det = dict([(k, v) for k, v in FROID.out_det_dict.items() if k in out_det])
        else:
            raise Exception(f"Unsupported type {type(out_det)} for out_det")

        if type(feat_red) == str and feat_red == "all":
            self.feat_red = FROID.feat_red_dict
        elif type(feat_red) == dict:
            self.feat_red = feat_red
        elif type(feat_red) == list:
            self.feat_red = dict([(k, v) for k, v in FROID.feat_red_dict.items() if k in feat_red])
        else:
            raise Exception(f"Unsupported type {type(out_det)} for out_det")

    def fit(self, X: np.ndarray):
        Xs_od = []
        Xs_fr = []

        input_od = input_fr = X

        for i in tqdm(range(self.iter), position=0, leave=False, desc="fit"):
            X_od, fitted_od = _run_od(input_od, self.out_det, self.kwargs)
            X_fr, fitted_fr = _run_fr(input_fr, self.feat_red, self.kwargs)

            self.fitted_ods.append(fitted_od)
            self.fitted_frs.append(fitted_fr)

            Xs_od.append(X_od)
            Xs_fr.append(X_fr)

            input_od = X_fr
            input_fr = X_od

        return np.hstack(Xs_od+Xs_fr)

    def transform(self, X: np.ndarray):
        Xs_od = []
        Xs_fr = []

        input_od = input_fr = X

        for i, fitted_od, fitted_fr in zip(tqdm(range(self.iter), position=0, leave=False, desc="transform"), self.fitted_ods,
                                           self.fitted_frs
                                           ):
            X_od, _ = _run_od(input_od, self.out_det, self.kwargs, fitted_od)
            X_fr, _ = _run_fr(input_fr, self.feat_red, self.kwargs, fitted_fr)

            Xs_od.append(X_od)
            Xs_fr.append(X_fr)

            input_od = X_fr
            input_fr = X_od

        return np.hstack(Xs_od + Xs_fr)



def _run_od(X, methods: dict(), kwargs: dict, fitted=None):

    X_out = np.zeros((len(X), len(methods))) * np.inf

    if fitted is None:
        fitted = []
        it = tqdm(methods.items(), position=1, leave=False)
        for i, (name, constructor) in enumerate(it):
            it.set_description(f"Fitting {name}")

            method = constructor(**{key: kwargs[key] for key in kwargs if key in constructor.__init__.__code__.co_varnames})
            method.fit(X)
            fitted.append(method)

            if isinstance(method, LocalOutlierFactor):
                X_out[:, i] = method.negative_outlier_factor_
            elif isinstance(method, EllipticEnvelope):
                X_out[:, i] = method.mahalanobis(X)
            elif "sklearn" in str(type(method)):
                X_out[:, i] = method.location_
            else:
                X_out[:, i] = method.decision_function(X)

    else:
        it = tqdm(zip(range(len(fitted)), methods.items(), fitted), total=len(fitted), position=1, leave=False)
        for i, (name, constructor), method in it:
            it.set_description(f"Running {name}")
            # method.predict(X)
            X_out[:, i] = method.decision_function(X)


    return X_out, fitted


def _run_fr(X, methods: dict(), kwargs: dict, fitted=None):
    X_out = []

    if fitted is None:
        fitted = []
        it = tqdm(methods.items(), position=1, leave=False)
        for name, constructor in it:
            it.set_description(f"Fitting {name}")

            method = constructor(**{key: kwargs[key] for key in kwargs if key in constructor.__init__.__code__.co_varnames})
            X_out.append(method.fit_transform(X))
            fitted.append(method)

    else:
        it = tqdm(zip(methods.items(), fitted), total=len(fitted), position=1, leave=False)
        for (name, constructor), method in it:
            it.set_description(f"Running {name}")
            X_out.append(method.transform(X))

    return np.hstack(X_out), fitted