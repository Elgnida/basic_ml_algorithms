import numpy as np
import matplotlib.pyplot as plt

class Parzen_Window_Classifier:

    def __init__(self, kernel_function='gaussian', h=1.0):

        self.h = h
        self.kernel_function = {
            'rectangle': lambda dist, h: (np.abs(dist/h) <= 1).astype(float),
            'epanechnikov': lambda dist, h: 0.75*(1 - (dist/h) ** 2) * (np.abs(dist/h) <= 1).astype(float),
            'triangle': lambda dist, h: (1 - np.abs(dist/h)) * (np.abs(dist/h) <= 1).astype(float),
            'gaussian': lambda dist, h: np.exp(-2*(dist/h) **2 ) / np.sqrt(2 * np.pi)
            }[kernel_function]

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.ravel() if np.ndim(y) == 2 else y
        self.classes = np.unique(self.y_train)

    def predict(self, X):

        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            weights = np.array([self.kernel_function(dist, self.h) for dist in distances])

            class_score = {}
            for cls in self.classes:
                class_score[cls] = np.sum(weights[self.y_train == cls])
            predict_class = max(class_score, key=class_score.get)
            predictions.append(predict_class)

        return np.array(predictions)

class Parzen_Window_Regression(Parzen_Window_Classifier):

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.ravel() if np.ndim(y) == 2 else y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            weights = np.array([self.kernel_function(dist, self.h) for dist in distances])
            predict = np.sum(weights * self.y_train)/np.sum(weights)
            predictions.append(predict)
        return np.array(predictions)


if __name__ == '__main__':

    n_samples = 100
    x_train = np.linspace(0, 4 * np.pi, n_samples).reshape(-1, 1)
    y_train = np.sin(x_train) + np.random.normal(loc=0.0, scale=0.5, size=x_train.shape)
    kernel_function = 'gaussian'
    h = 0.5
    model = Parzen_Window_Regression(kernel_function=kernel_function, h=h)
    model.fit(x_train, y_train)

    x_plot = np.linspace(0, 4 * np.pi, 500).reshape(-1, 1)
    y_pred = model.predict(x_plot)
    plt.figure(figsize=(12, 6))
    plt.scatter(x_train, y_train, color='red', s=20, alpha=0.6, label='Обучающие данные (с шумом)')
    plt.plot(x_plot, np.sin(x_plot), 'k--', linewidth=1.5, label='Истинная функция: $y = \sin(x)$')
    plt.plot(x_plot, y_pred, 'b-', linewidth=2.5, label=f'Parzen_Window-регрессия (h={h})')
    plt.title(f'Parzen Window-регрессия с {kernel_function} ядром')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
