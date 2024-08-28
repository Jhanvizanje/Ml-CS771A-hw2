import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

# Load the data from the provided pickle file
with open('mnist_small.pkl', 'rb') as file:
    data = pickle.load(file)

X = data['X']
Y = data['Y']

# Perform PCA to reduce the dimensionality to two dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform t-SNE to reduce the dimensionality to two dimensions
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
# colors = ['blue','green','red','y','orange','purple','k','c','m','brown']
def project(X_2d, cluster_labels, title):
    plt.figure(figsize=(10, 5))
    
    unique_clusters = np.unique(cluster_labels)
    for cluster_label in unique_clusters:
        indices = np.where(cluster_labels == cluster_label)[0]
        x_points = X_2d[indices, 0]
        y_points = X_2d[indices, 1]
        plt.scatter(x_points, y_points, marker='o', s=20, label=f'Cluster {cluster_label}', alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.show()



project(X_pca,Y,'PCA 2-D Projection')

project(X_tsne,Y,'t-SNE 2-D Projection')

