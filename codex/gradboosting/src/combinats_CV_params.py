# Utility script that I used when I launched the CV phase
# in parallel on FarmUI to accert no duplicates combinations
# of parameters would be taken into account.

import pickle

# Custom imports
from .utils import choose_CV_params

# Reasonable params that I extracted from 
# reading Scikit-learn documentation on them
possible_params = {
    'gamma': [0., 0.01, 0.1, 0.3],
    'lambda': [0.01, 0.1, 1., 5., 10., 100.],
    'learning_rate': [0.01, 0.03, 0.1, 0.3],
    'max_depth': [3, 4, 5, 6, 8, 10, 30],
    'max_features': [0.1, 0.25, 0.33, 0.5, 1.],
    'min_samples_leaf': [1],
    'min_samples_split': [2, 4],
    'min_split_gain': [0., 0.1, 0.3, 1.]    
}

# Generate 100 params combinations
combinats_CV_params = list(choose_CV_params(possible_params, 100))

with open('../CV_parameters.pkl', 'w') as f:
    pickle.dump(object, f)