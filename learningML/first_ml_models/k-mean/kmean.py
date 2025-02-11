import random
import numpy as np
from typing import Optional

class KMean:
    def __init__(self, k : int=2, max_iter: int=100, tol: float=0.0001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids: Optional[np.ndarray] = None
        self.clusters: Optional[np.ndarray] = None

    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Randomly selects k data points as initial centroids."""
        indices = np.random.choice(len(X), self.k, replace=False)
        return X[indices]

    def assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assigns each point to the nearest centroid."""
        """clusters = np.zeros(len(X), dtype=int)
        for i, point in enumerate(X):
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            clusters[i] = np.argmin(distances)
        return clusters"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X: np.ndarray) -> np.ndarray:
        """Updates centroids as the mean of assigned points"""
        return np.array([
            X[self.clusters == i].mean(axis=0) if np.any(self.clusters == i) else X[np.random.randint(len(X))]
            for i in range(self.k)
        ])

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Run the K-mean clustering algorithm."""
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            self.clusters = self.assign_clusters(X)
            new_centroids = self.update_centroids(X)

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        return self.clusters  # Fix: Return clusters
