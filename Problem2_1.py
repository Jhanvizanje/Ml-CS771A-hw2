import numpy as np
import matplotlib.pyplot as plt

# Load the data from the given text file
data = np.loadtxt('kmeans_data.txt')

# Visualize the original data
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1])
plt.title("Original Data")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# transforming the data by squaring them
transformed_data = np.square(data)

# Applying K-means algorithm on the transformed data
k = 2
np.random.seed(42)

# Randomly initializing the centroids
centroids = transformed_data[np.random.choice(transformed_data.shape[0], k, replace=False)]
max_iters = 100

for _ in range(max_iters):
    # Calculate distances between data points and centroids
    distances = np.linalg.norm(transformed_data - centroids[:, np.newaxis], axis=2)
    labels = np.argmin(distances, axis=0)
    new_centroids = np.array([transformed_data[labels == i].mean(axis=0) for i in range(k)])

    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

# Visualize the clustering results in the original 2D space
plt.figure(figsize=(6, 6))
cluster1 = data[labels == 0]
cluster2 = data[labels == 1]
plt.scatter(cluster1[:, 0], cluster1[:, 1], c='red', label='Cluster 1')
plt.scatter(cluster2[:, 0], cluster2[:, 1], c='green', label='Cluster 2')
plt.title("Clustering Results in original 2D space")
plt.xlabel("Feature-1")
plt.ylabel("Feature-2")
plt.show()
