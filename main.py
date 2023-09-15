import pandas as pd
import psutil
from sklearn.model_selection import train_test_split

from froid_od.classes.froid import FROID
from froid_od.classes.lfroid import LFROID

if __name__ == "__main__":
    df = pd.read_csv("datasets/bank.csv")

    X_train, X_test = train_test_split(df.values, test_size=.3, random_state=42)

    fr = FROID(seed=42, n_clusters=2, n_jobs=psutil.cpu_count())
    fr.fit(X_train)
    fr.transform(X_test)

    lfr = LFROID(seed=42, n_clusters=2)
    lfr.fit(X_train)
    lfr.transform(X_test)