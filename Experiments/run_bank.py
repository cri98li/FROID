import time

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, datasets

from froid_od.classes.adaboost_froid import adaboost_FROID
from froid_od.classes.bootstrap_froid import bootstrap_FROID
from froid_od.classes.froid import FROID
import wandb


"""df = pd.read_csv("../datasets/bank.csv").rename(columns={"TARGET": "target"})

X = df.drop(columns=["target"]).values
y = df[["target"]].values"""

(X, y) = datasets.load_iris(return_X_y=True)

X = RobustScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)


def run_bank():
    wandb.init(project="froid_demo")

    froid_i = FROID(
        seed=42, n_clusters=2, n_jobs=psutil.cpu_count(), froid_iter=2, verbose=False,
        n_components=1
    )

    fr = bootstrap_FROID(
        n_resamples=wandb.config.n_resamples,
        resample_rows_size=wandb.config.resample_rows_size,
        resample_cols_size=wandb.config.resample_cols_size,
        replace=wandb.config.replace,
        p=None,
        verbose=False,
        froid_base=froid_i,
        random_fr=wandb.config.random_fr,
        random_od=wandb.config.random_fr,
    )


    runtime = time.time()
    fr.fit(X_train)
    runtime = time.time() - runtime

    rand_for = RandomForestClassifier(n_jobs=psutil.cpu_count())
    rand_for.fit(np.hstack([fr.transform(X_train), X_train]), y_train)
    y_froid_predicted = rand_for.predict(np.hstack([fr.transform(X_test), X_test]))

    wandb.log({
        "training_time": runtime,
        "acc": metrics.accuracy_score(y_test, y_froid_predicted),
        "f1": metrics.f1_score(y_test, y_froid_predicted, average="macro"),
    })


wandb.agent("jjw99u99", project="froid_demo", function=run_bank, count=100)




