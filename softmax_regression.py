import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from utils import plot_decision_boundary
class SoftmaxRegression:

    def __init__(self, n_iteration=1000, learning_rate=0.1):
        self.n_iteration = n_iteration
        self.learning_rate = learning_rate

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        ohe_y = np.eye(n_classes)[y]
        self.b = np.zeros(n_classes)
        self.w = np.random.normal(size=(n_features, n_classes))
        for i in range(self.n_iteration):
            z = np.dot(X, self.w) + self.b
            probabilities = self.softmax_(z)
            CE = -np.sum(ohe_y * np.log(probabilities))
            print(f'Cross-Entropy loss is {CE:.2f} on {i + 1} iteration')
            dw = 1 / n_samples * X.T @ (probabilities - ohe_y)
            db = 1 / n_samples * np.sum((probabilities - ohe_y), axis=0)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        z = X @ self.w + self.b
        # print(self.softmax_(z))
        return np.argmax(self.softmax_(z), axis=1)

    def softmax_(self, z):
        proba = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return proba

iris = load_iris()
X_train = iris.data[:, [2, 3]]
y_train = iris.target

model = SoftmaxRegression()
model.fit(X_train, y_train)
preds = model.predict(X_train)
print(accuracy_score(y_train, preds))
plot_decision_boundary(model, X_train, y_train)
