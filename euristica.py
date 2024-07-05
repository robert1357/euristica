import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import numpy as np
from sklearn.neighbors import NearestNeighbors

def initialize_centroids(data, k):

    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def calculate_distance_matrix(data, centroids):

    return np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

def update_centroids_vectorized(data, assignments):

    new_centroids = np.array([data[assignments == i].mean(axis=0) for i in range(np.unique(assignments).size)])
    return new_centroids

def honeycomb_kmeans_optimized(data, k, d, gamma, omega):
    centroids = initialize_centroids(data, k)


    distance_matrix = calculate_distance_matrix(data, centroids)

    while True:
        knn = NearestNeighbors(n_neighbors=gamma)
        knn.fit(centroids)
        assignments = knn.kneighbors(data, return_distance=False)[:, 0]
        new_centroids = update_centroids_vectorized(data, assignments)
        distance = np.linalg.norm(centroids - new_centroids, axis=1).max()

        if distance < omega:
            break
        
        centroids = new_centroids

    return centroids, assignments

np.random.seed(42)
data = np.random.rand(100, 2)

k = 3
gamma = 1
omega = 0.01

centroids, assignments = honeycomb_kmeans_optimized(data, k, data.shape[1], gamma, omega)

print("Centroides:\n", centroids)
print("Asignaciones:\n", assignments)
