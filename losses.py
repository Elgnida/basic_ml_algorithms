import numpy as np

class L2_loss:
    def __call__(self, X, y, w, b, C):
        return np.sum((X @ w + b - y) ** 2) + C * np.sum(w ** 2)

    def grad(self, X, y, w, b, C):
        w_grad = (2 / X.shape[0]) * (X.T @ (np.dot(X, w) + b - y) + 2 * C * w)
        b_grad = (2 / X.shape[0]) * np.sum((X @ w) + b - y)
        return w_grad, b_grad

class L1_loss:
    def __call__(self, X, y, w, b, C):
        return np.sum((X @ w + b - y) ** 2) + C * np.sum(np.abs(w))

    def grad(self, X, y, w, b, C):
        w_grad = (2 / X.shape[0]) * (X.T @ (np.dot(X, w) + b - y)) + C * np.sign(w)
        b_grad = (2 / X.shape[0]) * np.sum((X @ w) + b - y)
        return w_grad, b_grad

class MSE_loss:
    def __call__(self, X, y, w, b, C):
        return np.sum((X @ w + b - y) ** 2)

    def grad(self, X, y, w, b, C):
        w_grad = (2 / X.shape[0]) * (X.T @ (np.dot(X, w) + b - y))
        b_grad = (2 / X.shape[0]) * np.sum((X @ w) + b - y)
        return w_grad, b_grad
