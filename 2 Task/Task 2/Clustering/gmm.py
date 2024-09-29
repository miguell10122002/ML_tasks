import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Load your data
X = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')

# Apply dimensionality reduction (PCA) to X
pca = PCA(n_components=2)  # Adjust the number of components as needed
X_pca = pca.fit_transform(X)

# Combine X and y into a single dataset
combined_data = np.column_stack((X_pca, y))

# Create a GMM model with 2 components (clusters)
gmm = GaussianMixture(n_components=2, random_state=0)

# Fit the model to the combined data
gmm.fit(combined_data)

# Get cluster assignments for each data point
cluster_assignments = gmm.predict(combined_data)

# Print samples in each cluster and the total number of samples
for cluster_id in range(2):
    cluster_samples = np.where(cluster_assignments == cluster_id)[0]
    total_samples = len(cluster_samples)
    print(f"Cluster {cluster_id + 1} Samples:", cluster_samples)
    print(
        f"Total Number of Samples in Cluster {cluster_id + 1}: {total_samples}")
