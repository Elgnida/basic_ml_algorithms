import numpy as np

def mse(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2).mean()

def mae(y_pred, y_true):
    return np.sum(np.abs(y_pred - y_true)).mean()
