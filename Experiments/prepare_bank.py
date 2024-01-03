import numpy as np
from pyod.models.knn import KNN

import wandb

sweep_config = {'method': 'grid'}
metric = {'name': 'acc', 'goal': 'maximize'}

sweep_config['metric'] = metric

parameters_dict = {
    'n_resamples': {
        'values': np.array(range(10, 100+1, 10)).tolist()
    },
    'resample_rows_size': {
        'values': (np.array(range(2, 5+1, 1))/10).tolist()
    },
    'resample_cols_size': {
        'values': (np.array(range(2, 5+1, 1))/10).tolist()
    },
    'replace': {
        'values': [True, False]
    },
    'random_fr': {
        'values': (np.array(range(1, 10, 2))/10).tolist()
    },
    'random_od': {
        'values': (np.array(range(1, 10, 2))/10).tolist()
    },
    'od': {
        'values': {"knn": KNN}
    },
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="froid_demo")

print(sweep_id)

