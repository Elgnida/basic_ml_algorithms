import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from metrics import mse
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class Decision_Tree_Regressor:
    def __init__(self, min_samples_leaf=2, max_depth=10, n_features=None):
        self.root = None
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        if (depth >= self.max_depth or n_samples <= self.min_samples_leaf):
            leaf_value = y.mean()
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = self._find_best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def _find_best_split(self, X, y, feat_idxs):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            threshold = np.mean(X_col)
            inf_gain = self._inf_gain(X_col, y, threshold)
            if inf_gain > best_gain:
                best_gain = inf_gain
                best_feature = feat_idx
                best_threshold = threshold

        return best_feature, best_threshold

    def _inf_gain(self, X_col, y, threshold):
        parent_pred = y.mean()
        parent_mse = self._mse(y, parent_pred)

        left_idxs, right_idxs = self._split(X_col, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(X_col)
        n_l, n_r = len(left_idxs), len(right_idxs)
        l_pred, r_pred = y[left_idxs].mean(), y[right_idxs].mean()
        l_mse, r_mse = self._mse(y[left_idxs], l_pred), self._mse(y[right_idxs], r_pred)
        child_mse = (n_l / n) * l_mse + (n_r / n) * r_mse
        information_gain = parent_mse - child_mse
        return information_gain

    def _mse(self, y_true, y_pred):
        return 1 / len(y_true) * np.sum((y_true - y_pred) ** 2)

    def _split(sefl, X_col, split_threshold):
        left_idxs = np.argwhere(X_col <= split_threshold).flatten()
        right_idxs = np.argwhere(X_col > split_threshold).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, X, node):
        if node.is_leaf_node():
            return node.value

        if X[node.feature_index] <= node.threshold:
            return self._traverse_tree(X, node.left)

        return self._traverse_tree(X, node.right)
