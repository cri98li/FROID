import copy

import numpy as np
from tqdm.auto import tqdm

from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.pca import PCA as pyod_PCA
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

from froid_od.classes.froid import FROID
from froid_od.classes.wrappers.loop import FROID_LocalOutlierProbability


class LFROID(FROID):

    out_det_dict = {
        "knn": KNN,
        "lof": LocalOutlierFactor,
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
        "pca": pyod_PCA,
    }

    feat_red_dict = {
        "pca": PCA,
        "isomap": Isomap,
        "kpca": KernelPCA,
        #"tsne": TSNE,
        #"rsp": GaussianRandomProjection,
    }
    def __init__(self, out_det="all", feat_red="all", froid_iter=1, seed=42, n_jobs=1, **kwargs):
        super().__init__(LFROID.out_det_dict, LFROID.feat_red_dict, froid_iter, seed, n_jobs, **kwargs)
