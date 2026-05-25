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
    
sigma1 = [[1, 0], [0, 1]]
sigma2 = [[10, 0], [0, 1]]
X0 = np.random.multivariate_normal([2, 2], sigma1, 200)
X1 = np.random.multivariate_normal([8, 3], sigma1, 200)
X2 = np.random.multivariate_normal([3, 6], sigma2, 200)
X = np.concatenate((X0, X1, X2), axis=0)
y = np.asarray([0]*200 + [1]*200 + [2]*200).T

model = KMeans(k_clusters=3)
labels_hst, centroids_hst, iters = model.fit(X)
print(f"Iters: {iters}")
print(f"Centroids: {centroids_hst[-1]}")

plt.figure(figsize=(10, 6))

# Vẽ dữ liệu
plt.scatter(X0[:, 0], X0[:, 1], color='blue', alpha=0.3, label='Cluster X0')
plt.scatter(X1[:, 0], X1[:, 1], color='green', alpha=0.3, label='Cluster X1')
plt.scatter(X2[:, 0], X2[:, 1], color='red', alpha=0.3, label='Cluster X2')

plt.scatter(centroids_hst[-1][:, 0], centroids_hst[-1][:, 1], 
            color='yellow', 
            marker='o', 
            s=250, 
            edgecolors='black', 
            linewidths=2, 
            label='Centroids')
plt.title('Biểu đồ phân cụm với các tâm (Centroids)', fontsize=14)
plt.xlabel('Trục $x_1$', fontsize=12)
plt.ylabel('Trục $x_2$', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Hiển thị
plt.tight_layout()
plt.show()

'''
Nhận xét: Bài 3 có số lần lặp lớn nhất. Nguyên nhân là dù số điểm của 3 cụm là bằng nhau nhưng cụm X2 có phân phối khác với 2 cụm còn lại, 
phương sai của trục x lớn gấp 10 lần trục y khiến cụm X2 có dạng bầu dục khi biểu diễn trên biểu đồ, khiến ranh giới giữa các cụm không rõ ràng, 
cụm X2 bị chia thành nhiều phần con trước khi được phân cụm đúng, làm cho số vòng lặp tăng đáng kể.
'''