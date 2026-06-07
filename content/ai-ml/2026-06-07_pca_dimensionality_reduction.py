"""
Principal Component Analysis (PCA) from Scratch
Dimensionality reduction via eigendecomposition.
"""
import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue descending
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_var
        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components.T + self.mean


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(200, 10)
    X[:, 1] = X[:, 0] * 2 + np.random.randn(200) * 0.1  # correlated features

    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Reduced shape:  {X_reduced.shape}")
    print(f"Explained variance ratios: {pca.explained_variance_ratio}")
    print(f"Total explained: {sum(pca.explained_variance_ratio):.4f}")
