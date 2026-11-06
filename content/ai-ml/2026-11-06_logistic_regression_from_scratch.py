"""
Logistic Regression from Scratch
Implementing binary classification with gradient descent.
"""
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = X @ self.weights + self.bias
            predictions = self._sigmoid(linear_pred)

            dw = (1 / n_samples) * (X.T @ (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_pred = X @ self.weights + self.bias
        y_pred = self._sigmoid(linear_pred)
        return (y_pred >= 0.5).astype(int)

    def predict_proba(self, X):
        linear_pred = X @ self.weights + self.bias
        return self._sigmoid(linear_pred)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"Accuracy: {accuracy:.4f}")
