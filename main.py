import time

import numpy as np
import pandas as pd
import psutil
from pyod.models.knn import KNN
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

from froid_od.classes.adaboost_froid import adaboost_FROID
from froid_od.classes.bagging_froid import bagging_FROID
from froid_od.classes.froid import FROID
from froid_od.classes.lfroid import LFROID
from froid_od.classes.bootstrap_froid import bootstrap_FROID

def base_classifier(X_train, X_test, y_train, y_test):
    rf_base = RandomForestClassifier(n_jobs=psutil.cpu_count())
    rf_base.fit(X_train, y_train)
    rf_base_predicted = rf_base.predict(X_test)
    print("Classification without froid")
    print(classification_report(y_test, rf_base_predicted))
def try_froid(X_train, X_test, y_train, y_test):
    od = ["knn", "lof", "cof", "mcd", "featurebagging", "copod", "isofor", "loda", "ocsvm", "pca"]
    fr = ["pca", "isomap", "kpca", "lle"]

    fr = FROID(seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=2, out_det=od, feat_red=fr, verbose=False)
    X_train_froid = fr.fit_transform(X_train)
    X_test_froid = fr.transform(X_test)

    rf_froid = RandomForestClassifier(n_jobs=psutil.cpu_count())
    rf_froid.fit(np.hstack([X_train_froid, X_train]), y_train)
    rf_froid_predicted = rf_froid.predict(np.hstack([X_test_froid, X_test]))
    print("Classification using base froid")
    print(classification_report(y_test, rf_froid_predicted))

def try_lfroid(X_train, X_test, y_train, y_test):
    fr = LFROID(seed=42, n_clusters=2, verbose=False)
    X_train_froid = fr.fit_transform(X_train)
    X_test_froid = fr.transform(X_test)

    rf_froid = RandomForestClassifier(n_jobs=psutil.cpu_count())
    rf_froid.fit(np.hstack([X_train_froid, X_train]), y_train)
    rf_froid_predicted = rf_froid.predict(np.hstack([X_test_froid, X_test]))
    print("Classification using L-froid")
    print(classification_report(y_test, rf_froid_predicted))

def try_bootstrap_froid(X_train, X_test, y_train, y_test):
    fr = bootstrap_FROID(
        n_resamples=100, resample_rows_size=.2, resample_cols_size=.2, replace=True, p=None, verbose=False,
        random_od=.1, random_fr=.1,
        froid_base=FROID(
            seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=1, verbose=False,
            n_components=2
        )
    )

    (X, y) = datasets.load_iris(return_X_y=True)
    X = RobustScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)

    X_train_froid = fr.fit_transform(X_train)
    X_test_froid = fr.transform(X_test)

    rf_froid = RandomForestClassifier(n_jobs=psutil.cpu_count())
    rf_froid.fit(np.hstack([X_train_froid, X_train]), y_train)
    rf_froid_predicted = rf_froid.predict(np.hstack([X_test_froid, X_test]))
    print("Classification using bootstrap froid")
    print(classification_report(y_test, rf_froid_predicted))

def try_bagging_froid(X_train, X_test, y_train, y_test):
    froid_i = FROID(
            seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=2, out_det=od, feat_red=fr, verbose=False,
            n_components=2
        )

    fr = bagging_FROID(
        froid=froid_i,
        estimator=RandomForestClassifier(n_jobs=psutil.cpu_count()),
        n_estimators=10
    )

    fr.fit(X_train, y_train)
    rf_froid_predicted = fr.predict(X_test)

    print("Classification using bagging froid")
    print(classification_report(y_test, rf_froid_predicted))

def try_adaboost_froid(X_train, X_test, y_train, y_test):
    froid_i = FROID(
        seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=1, verbose=True, n_components=2)

    fr = adaboost_FROID(
        froid=froid_i,
        estimator=DecisionTreeClassifier(),
        n_estimators=10
    )

    fr.fit(X_train, y_train)
    rf_froid_predicted = fr.predict(X_test)

    print("Classification using adaboost froid")
    print(classification_report(y_test, rf_froid_predicted))

def try_froid_dict(X_train, X_test, y_train, y_test):
    od = {
        "knn": KNN,
        "knn2": KNN(n_neighbors=2),
        "knn100": KNN(n_neighbors=100)
    }
    fr = {
        "pca2": PCA(n_components=2),
        "pca1": PCA(n_components=1)
    }

    fr = FROID(seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=2, out_det=od, feat_red=fr, verbose=False)
    X_train_froid = fr.fit_transform(X_train)
    X_test_froid = fr.transform(X_test)

    rf_froid = RandomForestClassifier(n_jobs=psutil.cpu_count())
    rf_froid.fit(np.hstack([X_train_froid, X_train]), y_train)
    rf_froid_predicted = rf_froid.predict(np.hstack([X_test_froid, X_test]))
    print("Classification using base froid with dict")
    print(classification_report(y_test, rf_froid_predicted))

if __name__ == "__main__":
    df = pd.read_csv("datasets/bank.csv").rename(columns={"TARGET": "target"})
    #df = pd.read_csv("datasets/generali2021.zip").fillna(-1)

    X = df.drop(columns=["target"]).values
    y = df[["target"]].values

    X = RobustScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)

    start = time.time()
    #base_classifier(X_train, X_test, y_train, y_test)
    print(f"{time.time() - start}\r\n\r\n")

    start = time.time()
    #try_froid_dict(X_train, X_test, y_train, y_test)
    print(f"{time.time() - start}\r\n\r\n")

    start = time.time()
    #try_froid(X_train, X_test, y_train, y_test)
    print(f"{time.time() - start}\r\n\r\n")

    start = time.time()
    #try_lfroid(X_train, X_test, y_train, y_test)
    print(f"{time.time() - start}\r\n\r\n")

    start = time.time()
    try_bootstrap_froid(X_train, X_test, y_train, y_test)
    print(f"{time.time() - start}\r\n\r\n")

    start = time.time()
    #try_bagging_froid(X_train, X_test, y_train, y_test)
    print(f"{time.time() - start}\r\n\r\n")

    start = time.time()
    #try_adaboost_froid(X_train, X_test, y_train, y_test)
    print(f"{time.time() - start}\r\n\r\n")

