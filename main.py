import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from froid_od.classes.froid import FROID
from froid_od.classes.lfroid import LFROID

if __name__ == "__main__":
    df = pd.read_csv("datasets/bank.csv")

    X = df.drop(columns=["TARGET"]).values
    y = df[["TARGET"]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
    od = [
        "knn", "lof", "loop", "cof", "hbos", "mcd", "featurebagging", "copod", "isofor", "loda", "ocsvm", "pca"
    ]

    fr = FROID(seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=5, out_det=od)
    X_train_froid = fr.fit(X_train)
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


    lfr = LFROID(seed=42, n_clusters=2)
    lfr.fit(X_train)
    lfr.transform(X_test)