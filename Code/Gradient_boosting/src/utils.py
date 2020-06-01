import random
import sys

import numpy as np

# Constant vars
NUMBER_Infinity = sys.maxsize 

class Dataset():
    """Helper class that stores both the input X
    and the output Y of some objective function f"""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

def calculate_MSE(Y, Y_pred):
    """MSE of the estimated output Y
    
    Parameters
    ----------
    Y_pred: np.ndarray
        Estimated prediction of Y

    Y: np.ndarray
        Original input Y

    """
    return np.mean(np.square(Y - Y_pred))

def choose_CV_params(params, num_choices):
    """Generator of num_choices random combinations
    of the given params to apply CV on."""
    
    picked_combinations = []
    
    for _ in range(num_choices):
        while True:
            combination = {}
            
            for p in params:
                combination[p] = np.random.choice(params[p])
            
            if combination not in picked_combinations:
                picked_combinations.append(combination)
                break
        
        yield combination

def generate_dataset_XY(n=10000, D=30):
    """Generate a synthetic dataset pair XY
    to test GBM for regression."""

    w = np.array([1] * D).reshape(D, 1)

    rand_min_D = [-0.1] * D
    rand_max_D = [0.1] * D

    AWGN_noise_sigma = 0.5
   
    X = np.zeros((n, D))

    for i in range(D):
        X[:, i] = np.random.uniform(rand_min_D[i],
                        rand_max_D[i], size=n)
    
    AWGN_noise = np.random.normal(0,
            AWGN_noise_sigma, size=(n,1))
    Y = np.dot(X, w)**2 + AWGN_noise
    
    return Dataset(X, Y)
