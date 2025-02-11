import random
import numpy as np

class KMean:
    def __init__(self, k=2, max_iter=100, tol=0.0001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.clusters = None

    def initialize_centroids(self, X):
        indices = random.sample(range(len(X)), self.k)
        return X[indices]

    def assign_clusters(self, X):
        """Assigns each point to the nearest centroid."""
        clusters = np.zeros(len(X), dtype=int)
        for i, point in enumerate(X):
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            clusters[i] = np.argmin(distances)
        return clusters

    def update_centroids(self, X):
        """Updates centroids as the mean of assigned points"""
        new_centroid = []
        for i in range(self.k):
            cluster_point = X[self.clusters == i]  # Fix: Proper indexing
            if len(cluster_point) > 0:
                new_centroid.append(cluster_point.mean(axis=0))
            else:
                new_centroid.append(X[random.randint(0, len(X) - 1)])  # Reinitialize if empty
        return np.array(new_centroid)

    def fit(self, X):
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
