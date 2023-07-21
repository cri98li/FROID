import copy

import numpy as np
from tqdm.auto import tqdm

from pyod.models.cblof import CBLOF
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


class FROID():

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
        "suod": SUOD,
        "ocsvm": OCSVM,
    }

    feat_red_dict = {
        "pca": PCA,
        "isomap": Isomap,
        #"mds": MDS,
        "kpca": KernelPCA,
        "tsne": TSNE,
        #"rsp": GaussianRandomProjection,
        "lle": LocallyLinearEmbedding,
        "se": SpectralEmbedding,
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

        if type(out_det) == str and out_det == "all":
            self.out_det = FROID.out_det_dict
        elif type(out_det) == dict:
            self.out_det = out_det
        elif type(out_det) == list:
            self.out_det = dict([(k, v) for k, v in out_det.items() if k in FROID.out_det_dict])
        else:
            raise Exception(f"Unsupported type {type(out_det)} for out_det")

        if type(feat_red) == str and feat_red == "all":
            self.feat_red = FROID.feat_red_dict
        elif type(feat_red) == dict:
            self.feat_red = feat_red
        elif type(feat_red) == list:
            self.feat_red = dict([(k, v) for k, v in feat_red.items() if k in FROID.feat_red_dict])
        else:
            raise Exception(f"Unsupported type {type(out_det)} for out_det")
