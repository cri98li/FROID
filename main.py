import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from froid_od.classes.froid import FROID
from froid_od.classes.lfroid import LFROID
from froid_od.classes.bootstrap_froid import bootstrap_FROID

def try_froid(X_train, X_test, y_train, y_test):
    od = ["knn", "lof", "cof", "mcd", "featurebagging", "copod", "isofor", "loda", "ocsvm", "pca"]
    fr = ["pca", "isomap", "kpca", "lle"]


    fr = FROID(seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=2, out_det=od, feat_red=fr, verbose=True)
    X_train_froid = fr.fit_transform(X_train)
    X_test_froid = fr.transform(X_test)

    rf_base = RandomForestClassifier()
    rf_base.fit(X_train, y_train)
    rf_base_predicted = rf_base.predict(X_test)
    print("Classification without froid")
    print(classification_report(y_test, rf_base_predicted))

    rf_froid = RandomForestClassifier()
    rf_froid.fit(np.hstack([X_train_froid, X_train]), y_train)
    rf_froid_predicted = rf_froid.predict(np.hstack([X_test_froid, X_test]))
    print("Classification using froid")
    print(classification_report(y_test, rf_froid_predicted))

def try_lfroid(X_train, X_test, y_train, y_test):
    fr = LFROID(seed=42, n_clusters=2, verbose=True)
    X_train_froid = fr.fit_transform(X_train)
    X_test_froid = fr.transform(X_test)

    rf_base = RandomForestClassifier()
    rf_base.fit(X_train, y_train)
    rf_base_predicted = rf_base.predict(X_test)
    print("Classification without froid")
    print(classification_report(y_test, rf_base_predicted))

    rf_froid = RandomForestClassifier()
    rf_froid.fit(np.hstack([X_train_froid, X_train]), y_train)
    rf_froid_predicted = rf_froid.predict(np.hstack([X_test_froid, X_test]))
    print("Classification using froid")
    print(classification_report(y_test, rf_froid_predicted))

def try_bootstrap_froid(X_train, X_test, y_train, y_test):
    od = ["knn", "lof"]
    fr = ["pca"]

    fr = bootstrap_FROID(
        n_resamples=100, resample_rows_size=.2, resample_cols_size=.2, replace=True, p=None, verbose=True,
        froid_base=FROID(
            seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=2, out_det=od, feat_red=fr, verbose=False,
            n_components=2
        )
    )

    X_train_froid = fr.fit_transform(X_train)
    X_test_froid = fr.transform(X_test)

    rf_base = RandomForestClassifier()
    rf_base.fit(X_train, y_train)
    rf_base_predicted = rf_base.predict(X_test)
    print("Classification without froid")
    print(classification_report(y_test, rf_base_predicted))

    rf_froid = RandomForestClassifier()
    rf_froid.fit(np.hstack([X_train_froid, X_train]), y_train)
    rf_froid_predicted = rf_froid.predict(np.hstack([X_test_froid, X_test]))
    print("Classification using froid")
    print(classification_report(y_test, rf_froid_predicted))


if __name__ == "__main__":
    df = pd.read_csv("datasets/bank.csv")

    X = df.drop(columns=["TARGET"]).values
    y = df[["TARGET"]].values

    X = RobustScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)

    #try_froid(X_train, X_test, y_train, y_test)
    #try_lfroid(X_train, X_test, y_train, y_test)
    try_bootstrap_froid(X_train, X_test, y_train, y_test)

