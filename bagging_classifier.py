from decision_tree_classifier import Decision_Tree_Classifier
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from metrics import accuracy
from utils import plot_decision_boundary
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

class Bagging_Classifier:

    def __init__(self, n_estimator=100, max_depth=10, n_samples=100):
        self.n_estimator = n_estimator
        self.estimators = []
        self.max_depth = max_depth
        self.n_samples = n_samples

    def fit(self, X, y):
        for _ in range(self.n_estimator):
            model = Decision_Tree_Classifier(max_depth=self.max_depth)
            idxs = self._bootstrap(X, self.n_samples)
            X_train = X[idxs]
            y_train = y[idxs]
            model.fit(X_train, y_train)
            self.estimators.append(model)

    def _bootstrap(self, X, n_samples):
        indicies = np.random.randint(0, len(X), size=n_samples)
        return indicies

    def predict(self, X):
        all_predictions = []
        for model in self.estimators:
            pred = model.predict(X)
            all_predictions.append(pred)
        all_predictions = np.array(all_predictions).T
        final_predictions = []
        for sample_predictions in all_predictions:
            hist = np.bincount(sample_predictions)
            pred = np.argmax(hist)
            final_predictions.append(pred)

        return np.array(final_predictions)

X_train, y_train = make_moons(noise=0.1, random_state=1)
tree = Decision_Tree_Classifier(max_depth=10)
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_train)

bagg_clf = Bagging_Classifier(max_depth=10)
bagg_clf.fit(X_train, y_train)
pred_bag = bagg_clf.predict(X_train)

sk_clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=10, min_samples_leaf=2))
sk_clf.fit(X_train, y_train)
pred_sk = sk_clf.predict(X_train)

acc_tree = accuracy(pred_tree, y_train)
bag_acc = accuracy(pred_bag, y_train)
sk_acc = accuracy(pred_sk, y_train)

print(f'bagging accurcy: {bag_acc}')
print(f'tree accuracy: {acc_tree}')
print(f'sklearn accuracy: {sk_acc}')
plot_decision_boundary(bagg_clf, X_train, y_train)
