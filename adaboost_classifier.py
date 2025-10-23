import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from utils import plot_decision_boundary
from decision_tree_classifier import Decision_Tree_Classifier
from sklearn.datasets import make_moons

class AdaBoost_Classifier:

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.model = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = len(X)
        weights = np.ones(n_samples) / n_samples
        self.classes = np.unique(y)
        for _ in range(self.n_estimators):
            #train weak classifier
            tree = Decision_Tree_Classifier(max_depth=1)
            tree.fit(X, y, sample_weights=weights)
            predictions = tree.predict(X)
            miss_class = (y != predictions)
            error = np.sum(weights[miss_class] / np.sum(weights))

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            weights[miss_class] *= np.exp(alpha)
            weights[~miss_class] *= np.exp(-alpha)
            weights /= np.sum(weights)

            self.model.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        final_predictions = []

        for i in range(len(X)):
            class_scores = {}

            for model, alpha in zip(self.model, self.alphas):
                predicted_class = model.predict([X[i]])[0]
                if predicted_class in class_scores:
                    class_scores[predicted_class] += alpha
                else:
                    class_scores[predicted_class] = alpha
            best_class = max(class_scores.items(), key=lambda x: x[1])[0]
            final_predictions.append(best_class)

        return np.array(final_predictions)


if __name__ == '__main__':
    X_train, y_train = make_moons(noise=0.1, random_state=1)
    model = AdaBoost_Classifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    plot_decision_boundary(model, X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_train, preds):.4f}")
