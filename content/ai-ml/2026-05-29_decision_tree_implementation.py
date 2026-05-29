"""
Decision Tree Classifier (CART Algorithm)
Binary splits using Gini impurity.
"""
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf prediction

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        counts = Counter(y)
        n = len(y)
        return 1.0 - sum((c/n)**2 for c in counts.values())

    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        return X[left_mask], X[~left_mask], y[left_mask], y[~left_mask]

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self._gini(y)
        n = len(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                _, _, y_left, y_right = self._split(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = parent_gini - (len(y_left)/n)*self._gini(y_left) - (len(y_right)/n)*self._gini(y_right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feature, threshold

        return best_feat, best_thresh

    def _build(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return Node(value=Counter(y).most_common(1)[0][0])

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        X_left, X_right, y_left, y_right = self._split(X, y, feat, thresh)
        return Node(
            feature=feat, threshold=thresh,
            left=self._build(X_left, y_left, depth+1),
            right=self._build(X_right, y_right, depth+1),
        )

    def fit(self, X, y):
        self.root = self._build(np.array(X), np.array(y))

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return [self._predict_one(x, self.root) for x in np.array(X)]
