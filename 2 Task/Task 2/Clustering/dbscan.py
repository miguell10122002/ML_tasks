import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load your data (replace with your data loading code)
X = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')

# Standardize the feature data (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Concatenate feature data and target data horizontally
data_with_y = np.column_stack((X_scaled, y))

# Initialize the DBSCAN model
eps = 2  # Radius for neighborhood search
min_samples = 5  # Minimum number of samples in a neighborhood
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the DBSCAN model to your combined data
dbscan.fit(data_with_y)

# Extract cluster labels and identify outliers (-1 label)
cluster_labels = dbscan.labels_
outliers = data_with_y[cluster_labels == -1]

# Visualize the results (2D case, adapt for your dataset)
if X_scaled.shape[1] == 2:
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
                c=cluster_labels, cmap='viridis', marker='o', s=50)
    plt.scatter(outliers[:, 0], outliers[:, 1], c='red',
                marker='x', s=100, label='Outliers')
    plt.title('DBSCAN Clustering with Outlier Detection (including y)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Print the number of outliers and their indices
num_outliers = len(outliers)
print(f"Number of Outliers: {num_outliers}")
if num_outliers > 0:
    outlier_indices = np.where(cluster_labels == -1)[0]
    print("Indices of Outliers:", outlier_indices)
