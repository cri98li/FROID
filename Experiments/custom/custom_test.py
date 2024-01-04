OD = {
    "LoOP": {
        "k": [1, 5, 10, 20]
    },
    "COF": {
        "k": [1, 5, 10, 20]
    },
    "KNN": {
        "k": [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250],
        "method": ["mean", "median", "largest"]
    },
    "LOF": {
        "k": [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250]
    },
    "IsoFor": {
        "n": [0, 20, 50, 70, 100, 150, 200, 250],
    },
    "ALL": {
        "contamination": [.001, .01, .1, .2, .5],
        "extent": [1,2,3],
        "v": [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
    }
}

FR = {
    "ALL": {
        "kernel": ["rbf", "sigmoid", "cosine", "poly"],
    },
    "MDS": {
        "max_iter": [100]
    },
    "LLE": {
        "method": ["hessian", "modified"]
    }
}