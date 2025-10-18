import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

#SVM with soft-margin
class SVM:

    def __init__(self, lr = 0.001, C = 0.01, n_iterations = 1000):
        self.lr = lr
        self.C = C
        self.n_iterations = n_iterations
        self.w = None
        self.b = None


    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for i, Xi in enumerate(X):
                if y[i] * (np.dot(Xi, self.w) - self.b) >= 1 :
                    self.w -= self.lr * (2 * self.C * self.w)
                else:
                    self.w -= self.lr * (2 * self.C * self.w - np.dot(Xi, y[i]))
                    self.b -= self.lr * y[i]

    def predict(self, X):
        pred = np.dot(X, self.w) - self.b
        result = [1 if val > 0 else -1 for val in pred]
        return result

if __name__ == "__main__":
    X_train, y_train = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2.)
    y_train = y_train * 2 - 1

    svm = SVM()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_train)
    print(f"accuracy: {accuracy_score(y_train, y_pred) * 100}%")

    w = svm.w
    b = svm.b
    print(w, b)
    x_disp = np.linspace(np.min(X_train[:,0]), np.max(X_train[:,0]), num=10)
    #optimal hyperplane
    y = lambda x: (-x * w[0] + b) / w[1]
    y_disp = [y(x) for x in x_disp]
    plt.plot(x_disp, y_disp, 'red', label='SVM')
    #first edge of the hyperplane
    y = lambda x: (-x * w[0] - 1 + b) / w[1]
    y_disp = [y(x) for x in x_disp]
    plt.plot(x_disp, y_disp, 'red', label='edge', linestyle=':', linewidth=0.5)
    #second edge of the hyperplane
    y = lambda x: (-x * w[0] + 1 + b) / w[1]
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
