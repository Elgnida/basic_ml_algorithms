import numpy as np
import matplotlib.pyplot as plt
from metrics import mse, mae
from losses import L2_loss, L1_loss, MSE_loss

class LinearRegression:

    def __init__(self, n_iteration=100, learning_rate=0.01, penalty='l1', C=0):
        self.n_iteration = n_iteration
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.C = C
        self.loss_function = {None: MSE_loss(),
                              'l2': L2_loss(),
                              'l1': L1_loss()}[penalty]


    def fit(self, X, y):
        n_features = X.shape[1]
        self.w = np.random.normal(loc=0.0, scale=0.1, size=(n_features, 1))
        self.b = np.random.normal(loc=0.0, scale=0.1)
        self.losses = []
        for _ in range(self.n_iteration):
            y_pred = (X @ self.w) + self.b
            self.losses.append(mae(y_pred, y))
            w_grad, b_grad = self.loss_function.grad(X, y, self.w, self.b, self.C)
            self.w -= self.learning_rate * w_grad
            self.b -= self.learning_rate * b_grad

    def predict(self, X):
        return np.dot(X, self.w) + self.b

if __name__ == '__main__':
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    model = LinearRegression(penalty=None)
    model.fit(X, y)

    X_new = np.array([[0], [2]])
    y_new = model.predict(X_new)

    plt.plot(X_new, y_new, "r-")
    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.show()
