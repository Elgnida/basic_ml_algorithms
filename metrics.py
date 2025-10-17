import numpy as np

def mse(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2).mean()

def mae(y_pred, y_true):
    return np.sum(np.abs(y_pred - y_true)).mean()

def accuracy(y_pred, y_true):
    return np.sum(np.equal(y_true, y_pred)) / y_true.size
