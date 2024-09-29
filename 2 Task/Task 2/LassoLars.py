import numpy as np
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LassoLars
from sklearn.model_selection import LeavePOut
from sklearn.metrics import mean_squared_error

import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Load your data
X = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')

# Combine X and y into a single dataset
combined_data = np.column_stack((X, y))

# Create a GMM model with 2 clusters
gmm = GaussianMixture(n_components=2, random_state=None)

# Fit the model to the data
gmm.fit(combined_data)

# Get cluster assignments
gmm_labels = gmm.predict(combined_data)

# Print the cluster assignments in a descriptive format
print("\nGMM Clusters:")
cluster_0_samples_gmm = np.where(gmm_labels == 0)[0]
cluster_1_samples_gmm = np.where(gmm_labels == 1)[0]
print(
    f"Cluster 0 (GMM): Samples {cluster_0_samples_gmm} ({len(cluster_0_samples_gmm)} samples)")
print(
    f"Cluster 1 (GMM): Samples {cluster_1_samples_gmm} ({len(cluster_1_samples_gmm)} samples)\n")

# Create two subsets based on the clusters
subset_0 = combined_data[gmm_labels == 0]
subset_1 = combined_data[gmm_labels == 1]

# Separate the subsets into features (X) and target (y)
X_subset_0 = subset_0[:, :-1]  # Exclude the last column which is the target
y_subset_0 = subset_0[:, -1]   # The last column is the target

X_subset_1 = subset_1[:, :-1]
y_subset_1 = subset_1[:, -1]

# Define a function for LassoLars with LeavePOut


def perform_lasso_lars(X, y, sample_counts_to_leave_out):
    alpha_values = np.arange(0, 50, 0.1)

    # Create the LeavePOut cross-validator
    lpo = LeavePOut(p=sample_counts_to_leave_out)

    # Initialize arrays to store MSE values for different alphas
    mse_values = np.zeros(len(alpha_values))

    # Perform LassoLars with LeavePOut
    for i, alpha in enumerate(alpha_values):
        mse_for_alpha = 0

        for train_index, test_index in lpo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            lasso_lars = LassoLars(alpha=alpha)
            lasso_lars.fit(X_train, y_train)

            y_pred = lasso_lars.predict(X_test)
            mse_for_alpha += mean_squared_error(y_test, y_pred)

        mse_values[i] = mse_for_alpha / len(X)

    # Find the index of the optimal alpha with minimum MSE
    optimal_alpha_index = np.argmin(mse_values)
    optimal_alpha = alpha_values[optimal_alpha_index]

    print(f"Optimal Alpha:", optimal_alpha)

    # Print the Mean MSE (Cross-Validation)
    best_mse = mse_values[optimal_alpha_index]
    print("Mean MSE (Cross-Validation):", best_mse)

    final_lasso_lars = LassoLars(alpha=optimal_alpha)
    final_lasso_lars.fit(X, y)

    # Print R-squared (R²) for the model
    y_predicted = final_lasso_lars.predict(X)
    r_squared = r2_score(y, y_predicted)
    rounded_r2 = "%.3f" % r_squared
    print("R-squared (R²) Score for the model:", rounded_r2)

    # Print the final SSE
    sse = np.linalg.norm((y - y_predicted) ** 2)
    print("The SSE is:", sse)

    return final_lasso_lars

# Now you have two subsets: X_subset_0, y_subset_0, and X_subset_1, y_subset_1
# You can use these subsets to train two different models based on two different datasets.


sample_counts_to_leave_out = 1

# Perform LassoLars with LeavePOut for the first subset
print("Values for Subset 0 with 1 sample left out:")
model_0 = perform_lasso_lars(
    X_subset_0, y_subset_0, sample_counts_to_leave_out)

print("---------------------------")

# Perform LassoLars with LeavePOut for the second subset
print("Values for Subset 1 with 1 sample left out:")
model_1 = perform_lasso_lars(
    X_subset_1, y_subset_1, sample_counts_to_leave_out)

# You now have two models, model_0 and model_1, trained on the respective subsets.
