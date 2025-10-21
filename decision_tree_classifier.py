import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from utils import plot_decision_boundary

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class Decision_Tree_Classifier:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape

        # Защита от пустых данных
        if n_samples == 0:
            return Node(value=0)

        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or
            n_samples <= self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = self._find_best_split(X, y, feat_idxs)

        # Если не нашли хорошее разделение
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def _find_best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                inf_gain = self._inf_gain(y, X_col, threshold)

                if inf_gain > best_gain:
                    best_gain = inf_gain
                    split_idx = feat_idx
                    split_threshold = threshold

        # Если информационный прирост не положительный
        if best_gain <= 0:
            return None, None

        return split_idx, split_threshold

    def _inf_gain(self, y, X_col, threshold):
        parent_entropy = self._entropy(y)
        left_idx, right_idx = self._split(X_col, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_col, split_threshold):
        left_idxs = np.argwhere(X_col <= split_threshold).flatten()
        right_idxs = np.argwhere(X_col > split_threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0  # или можно вернуть случайный класс
        counter = Counter(y)
        value = counter.most_common(1)[0][0]  # Исправлено здесь!
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, X, node):
        if node.is_leaf_node():
            return node.value

        if X[node.feature] <= node.threshold:
            return self._traverse_tree(X, node.left)

        return self._traverse_tree(X, node.right)

if __name__ == '__main__':
    iris = load_iris()
    X_train = iris.data[:, [0, 1]]
    y_train = iris.target

    model = Decision_Tree_Classifier(min_samples_split=3, max_depth=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    plot_decision_boundary(model, X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_train, preds):.4f}")
