import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

#SVM with soft-margin
class SVM:
    def __init__(self, n_iter=1000, C=1.0, lr=0.001, random_state=None):
        self.n_iter = n_iter
        self.C = C
        self.lr = lr
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        n, m = X.shape
        self.w = np.random.normal(0, 0.01, m)
        self.b = 0.0
        self.loss = []

        for _ in range(self.n_iter):
            self.loss.append(self.cost_func(X, y))
            for i in range(n):
                margin = y[i] * (np.dot(X[i], self.w) + self.b)
                if margin >= 1:
                    w_grad = self.w
                    b_grad = 0
                else:
                    w_grad = self.w - self.C * y[i] * X[i]
                    b_grad = -self.C * y[i]

                self.w -= self.lr * w_grad
                self.b -= self.lr * b_grad

    def cost_func(self, X, y):
        margin = y * (np.dot(X, self.w) + self.b)
        hinge_losses = np.maximum(0, 1 - margin)
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_losses)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

if __name__ == "__main__":
    X_train, y_train = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2.)
    y_train = y_train * 2 - 1

    svm = SVM()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_train)
    print(f"accuracy: {accuracy_score(y_train, y_pred) * 100}%")

    w = svm.w
    b = svm.b

    x_disp = np.linspace(np.min(X_train[:,0]), np.max(X_train[:,0]), num=10)
    #optimal hyperplane
    y = lambda x: -(x * w[0] + b) / w[1]
    y_disp = [y(x) for x in x_disp]
    plt.plot(x_disp, y_disp, 'red', label='SVM')
    #first edge of the hyperplane
    y = lambda x: -(x * w[0] - 1 + b) / w[1]
    y_disp = [y(x) for x in x_disp]
    plt.plot(x_disp, y_disp, 'red', label='edge', linestyle=':', linewidth=0.5)
    #second edge of the hyperplane
    y = lambda x: -(x * w[0] + 1 + b) / w[1]
    y_disp = [y(x) for x in x_disp]
    plt.plot(x_disp, y_disp, 'red', label='edge', linestyle=':', linewidth=0.5)
    #plot Classification decision boundary
    plt.title("Support Vector Machine")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(X_train[y_train == 1][:,0], X_train[y_train == 1][:,1], marker='_',color='blue', label='cluster 1')
    plt.scatter(X_train[y_train == -1][:,0], X_train[y_train == -1][:,1], marker='+',color='green',  label='cluster 2')
    plt.legend(loc=2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.show()
