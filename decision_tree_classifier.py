import numpy as np
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

    def fit(self, X, y, sample_weights=None):

        if sample_weights is None:
            self.sample_weights = np.ones(len(y))
        else:
            self.sample_weights = sample_weights
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y, self.sample_weights)

    def _grow_tree(self, X, y, weights, depth=0):

        n_samples, n_feats = X.shape
        if n_samples == 0:
            return Node(value=0)

        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or
            n_samples <= self.min_samples_split):
            leaf_value = self._most_common_label(y, weights)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = self._find_best_split(X, y, feat_idxs, weights)

        if best_feature is None:
            leaf_value = self._most_common_label(y, weights)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], weights[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], weights[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)

    def _find_best_split(self, X, y, feat_idxs, weights):

        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                inf_gain = self._inf_gain(y, X_col, threshold, weights)
                if inf_gain > best_gain:
                    best_gain = inf_gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _inf_gain(self, y, X_col, threshold, weights):

        left_idx, right_idx = self._split(X_col, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        parent_weight_gini = self._weighted_gini(y, weights)
        l_w, r_w = weights[left_idx], weights[right_idx]
        if len(l_w) == 0 or len(r_w) == 0:
            return 0

        l_g, r_g = self._weighted_gini(y[left_idx], l_w), self._weighted_gini(y[right_idx], r_w)
        information_gain = parent_weight_gini - (np.sum(l_w) * l_g / np.sum(weights) + np.sum(r_w) * r_g / np.sum(weights))
        return information_gain

    def _weighted_gini(self, y, weights):

        classes = np.unique(y)
        total_weight = np.sum(weights)
        weighted_hist = []

        for cls in classes:
            cls_weight = np.sum(weights[y == cls])
            weighted_hist.append(cls_weight / total_weight)

        weighted_hist = np.array(weighted_hist)
        return 1 - np.sum(weighted_hist ** 2)

    def _split(self, X_col, split_threshold):

        left_idxs = np.argwhere(X_col <= split_threshold).flatten()
        right_idxs = np.argwhere(X_col > split_threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y, weights):

        if len(y) == 0:
            return 0

        unique_classes = np.unique(y)
        best_class = unique_classes[0]
        max_weight = 0
        for cls in unique_classes:
            cls_weight = np.sum(weights[y == cls])
            if cls_weight > max_weight:
                max_weight = cls_weight
                best_class = cls
        return best_class

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
    X = iris.data[:, [0, 1]]
    y = iris.target
    mask = y != 2
    X_train = X[mask]
    y_train = y[mask]
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    sample_weights = np.ones(len(X_train)) / len(X_train)
    model = Decision_Tree_Classifier(min_samples_split=2, max_depth=5)
    model.fit(X_train, y_train, sample_weights=sample_weights)
    preds = model.predict(X_train)
    plot_decision_boundary(model, X_train, y_train)
    print(f"Accuracy: {accuracy_score(y_train, preds):.4f}")
