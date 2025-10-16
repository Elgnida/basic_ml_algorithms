import numpy as np

def mse(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2) / y_true.size
