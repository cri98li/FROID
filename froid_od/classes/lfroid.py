from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.mcd import MCD
from pyod.models.pca import PCA as pyod_PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import LocalOutlierFactor

from froid_od.classes.froid import FROID


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
        #"suod": SUOD, # problemi di versioni
        "pca": pyod_PCA,
    }

    feat_red_dict = {
        "pca": PCA,
        "isomap": Isomap,
        "kpca": KernelPCA,
        #"tsne": TSNE, #aggiungere il metodo transforms
        #"rsp": GaussianRandomProjection, # vedere gli iperparametri
    }
    def __init__(self, froid_iter=1, seed=42, n_jobs=1, **kwargs):
        super().__init__(LFROID.out_det_dict, LFROID.feat_red_dict, froid_iter, seed, n_jobs, **kwargs)
