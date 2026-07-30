"""
K-Means Clustering Implementation
Partitioning data into K clusters by minimizing within-cluster variance.
"""
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.RandomState(random_state)
        self.centroids = None
        self.labels = None
        self.inertia = None

    def _init_centroids(self, X):
        """K-Means++ initialization for better convergence."""
        n_samples = X.shape[0]
        centroids = [X[self.rng.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            dists = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
            probs = dists / dists.sum()
            idx = self.rng.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            # Assign clusters
            distances = np.array([
                np.sum((X - c)**2, axis=1) for c in self.centroids
            ]).T
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[self.labels == k].mean(axis=0)
                if np.sum(self.labels == k) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            shift = np.sum((new_centroids - self.centroids)**2)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.inertia = sum(
            np.sum((X[self.labels == k] - self.centroids[k])**2)
            for k in range(self.n_clusters)
        )
        return self

    def predict(self, X):
        distances = np.array([
            np.sum((X - c)**2, axis=1) for c in self.centroids
        ]).T
        return np.argmin(distances, axis=1)


def elbow_method(X, max_k=10):
    """Find optimal K using the elbow method."""
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k)
        km.fit(X)
        inertias.append(km.inertia)
    return inertias


if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(50, 2) + [2, 2],
        np.random.randn(50, 2) + [-2, -2],
        np.random.randn(50, 2) + [2, -2],
    ])

    km = KMeans(n_clusters=3)
    km.fit(X)
    print(f"Centroids:\n{km.centroids}")
    print(f"Inertia: {km.inertia:.2f}")
    print(f"Cluster sizes: {[np.sum(km.labels == k) for k in range(3)]}")
