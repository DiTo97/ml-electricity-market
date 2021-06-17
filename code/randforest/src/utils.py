import numpy as np

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
