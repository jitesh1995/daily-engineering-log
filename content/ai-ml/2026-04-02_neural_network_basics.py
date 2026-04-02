"""
Simple Feed-Forward Neural Network
Two-layer network with backpropagation for binary classification.
"""
import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim=16, lr=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = (1/m) * (self.a1.T @ dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_deriv(self.z1)
        dW1 = (1/m) * (X.T @ dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=500):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = -np.mean(y.reshape(-1,1)*np.log(output+1e-8) +
                           (1-y.reshape(-1,1))*np.log(1-output+1e-8))
            self.backward(X, y)
            if epoch % 100 == 0:
                losses.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses
