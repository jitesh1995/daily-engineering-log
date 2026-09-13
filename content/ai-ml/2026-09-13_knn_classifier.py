"""
K-Nearest Neighbors Classifier
Distance-based classification with support for different metrics.
"""
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric

    def _distance(self, a, b):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b))
        elif self.metric == "cosine":
            return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    model = KNNClassifier(k=5, metric="euclidean")
    model.fit(X_train, y_train)

    X_test = np.random.randn(20, 2)
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
