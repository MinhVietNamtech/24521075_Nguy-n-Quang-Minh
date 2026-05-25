import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# K-Means
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

class KMeans:
    def __init__(self, k_clusters=3):
        self.k_clusters = k_clusters
        self.labels_hst = []
        self.centroids_hst = []

    def _init_centroids(self):
        return self.X[np.random.choice(self.X.shape[0], self.k_clusters)]
    def _get_centroids(self, labels):
        centroids = np.zeros((self.k_clusters, self.X.shape[1]))

        for i in range(self.k_clusters):
            points = self.X[labels == i, :]
            centroids[i, :] = np.mean(points, axis=0)
        return centroids
    def _assign_labels(self, centroids):
        labels = np.zeros(self.X.shape[0])
        for idx, x in enumerate(self.X):
            dist = [euclidean_distance(x, centroid) for centroid in centroids]
            labels[idx] = np.argmin(dist)
        return labels
    def _has_converged(self, old_centroids, new_centroids):
        return (set([tuple(a) for a in old_centroids]) == set([tuple(b) for b in new_centroids]))
    
    def fit(self, X):
        self.X = X
        self.centroids_hst.append(self._init_centroids())
        it = 0
        while True:
            labels = self._assign_labels(self.centroids_hst[-1])
            new_centroids = self._get_centroids(labels)
            if self._has_converged(self.centroids_hst[-1], new_centroids) == True:
                break
            self.labels_hst.append(labels)
            self.centroids_hst.append(new_centroids)
            it += 1
        return self.labels_hst, self.centroids_hst, it
    
sigma = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal([2, 2], sigma, 200)
X1 = np.random.multivariate_normal([8, 3], sigma, 200)
X2 = np.random.multivariate_normal([3, 6], sigma, 200)
X = np.concatenate((X0, X1, X2), axis=0)
y = np.asarray([0]*200 + [1]*200 + [2]*200).T

model = KMeans(k_clusters=3)
labels_hst, centroids_hst, iters = model.fit(X)
print(f"Iters: {iters}")
print(f"Centroids: {centroids_hst[-1]}")

# Nhận xét: Việc khởi tạo centroids ngẫu nhiên ảnh hưởng đến tốc độ hội tụ của thuật toán