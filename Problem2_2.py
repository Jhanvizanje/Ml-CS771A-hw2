import numpy as np
import matplotlib.pyplot as plt

# Loading the data
input_data = np.loadtxt('kmeans_data.txt')

# Implementing K-means
def custom_kmeans(X, num_clusters, max_itr=5, random_seed=None):
    np.random.seed(random_seed)
    
    # Randomly initialize the centroids
    centroids = X[np.random.choice(X.shape[0], num_clusters, replace=False)]
    
    for _ in range(max_itr):
        # Calculate distances from each point to each centroid, assign to the closest one, and then update them
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        for i in range(num_clusters):
            centroids[i] = np.mean(X[labels == i], axis=0)
    
    return labels, centroids

num_runs = 10

# Define the RBF kernel function
def radial_basis_function_kernel(X, landmark, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(X - landmark, axis=1)**2)

# Perform 10 runs with different randomly chosen landmarks
for run in range(num_runs):
    # Randomly choose a single landmark and compute feature using RBF kernel
    landmark_index = np.random.choice(len(input_data))
    chosen_landmark = input_data[landmark_index]
    landmark_feature = radial_basis_function_kernel(input_data, chosen_landmark)

    # Apply K-means on the single landmark-based feature
    cluster_labels, cluster_centroids = custom_kmeans(landmark_feature.reshape(-1, 1), num_clusters=2, random_seed=run)

    # Plot the clustering results and the chosen landmark
    cluster1_points = input_data[cluster_labels == 0]
    cluster2_points = input_data[cluster_labels == 1]
    
    plt.scatter(cluster1_points[:, 0], cluster1_points[:, 1], marker='o', color='red', label='Cluster 1')
    plt.scatter(cluster2_points[:, 0], cluster2_points[:, 1], marker='o', color='green', label='Cluster 2')
    
    plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 0], marker='o', color='black', label='Centroids')

    plt.scatter(chosen_landmark[0], chosen_landmark[1], marker='x', color='blue', label='Chosen Landmark')
    plt.title(f'Run {run + 1} - K-means Clustering Landmark')
    plt.legend()
    plt.show()
