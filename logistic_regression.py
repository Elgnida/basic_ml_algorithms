import numpy as np
from losses import LogLoss
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from utils import plot_decision_boundary

class Logistic_Regression:

    def __init__(self, learning_rate=0.01, n_iteration=1000):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.loss_func = LogLoss()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        y = y.reshape(-1, 1)
        n_features = X.shape[1]
        self.w = np.random.normal(loc=0.0, scale=0.1, size=(n_features, 1))
        self.losses = []
        for _ in range(self.n_iteration):
            decision_func = X @ self.w
            proba = self.sigmoid(decision_func)
            loss = self.loss_func(y, proba)
            self.losses.append(loss)
            dw = self.loss_func.grad(X, y, proba)
            self.w -= self.learning_rate * dw

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        proba = self.sigmoid(X @ self.w)
        result = np.where(proba >= 0.5, 1, 0)
        return result.ravel()

if __name__ == '__main__':
    X_train, y_train = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2.)
    model = Logistic_Regression()
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    print(f'Acuracy score {accuracy_score(y_train, preds) * 100}%')
    plot_decision_boundary(model, X_train, y_train)
