import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeavePOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
    f"Cluster 1 (GMM): Samples {cluster_1_samples_gmm} ({len(cluster_1_samples_gmm)} samples)")

# Create two subsets based on the clusters
subset_0 = combined_data[gmm_labels == 0]
subset_1 = combined_data[gmm_labels == 1]

# Separate the subsets into features (X) and target (y)
X_subset_0 = subset_0[:, :-1]  # Exclude the last column which is the target
y_subset_0 = subset_0[:, -1]   # The last column is the target

X_subset_1 = subset_1[:, :-1]
y_subset_1 = subset_1[:, -1]

# Define a function for LassoCV with LeavePOut


def perform_lasso_cv(X, y, sample_counts_to_leave_out, y_scaler):
    # Start from a small positive alpha
    alpha_values = np.arange(0.001, 50, 0.1)

    # Create the LeavePOut cross-validator
    lpo = LeavePOut(p=sample_counts_to_leave_out)

    # Initialize a StandardScaler to normalize features
    scaler = StandardScaler()

    best_alpha = 0
    best_mse = 10

    for alpha in alpha_values:
        mse_scores = []

        for train_index, test_index in lpo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Normalize features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Normalize target variable y (use the same y_scaler)
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

            # Create and train the Lasso model
            lasso = Lasso(alpha=alpha)
            lasso.fit(X_train_scaled, y_train_scaled)

            # Predict
            y_pred_scaled = lasso.predict(X_test_scaled)

            # Reshape y_pred_scaled to a 2D array before denormalization
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)

            # Denormalize y_pred
            y_pred = y_scaler.inverse_transform(y_pred_scaled)

            # Calculate MSE
            mse = mean_squared_error(y_test, y_pred)

            mse_scores.append(mse)

        if np.mean(mse_scores) < best_mse:
            best_mse = np.mean(mse_scores)
            best_alpha = alpha

    print(f"Optimal Alpha:", best_alpha)
    print("MSE (Cross-Validation):", best_mse)

    final_lasso_reg = Lasso(alpha=best_alpha)

    # Normalize features in the entire dataset X
    X_scaled = scaler.fit_transform(X)

    # Normalize target variable y (use the same y_scaler)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

    final_lasso_reg.fit(X_scaled, y_scaled)

    # Predict on the normalized data
    y_predicted_scaled = final_lasso_reg.predict(X_scaled)

    # Inverse transform the predictions to get them in the original scale
    y_predicted = y_scaler.inverse_transform(y_predicted_scaled)

    # Calculate R-squared (R²) for the model
    r_squared = r2_score(y, y_predicted)

    # Calculate SSE for the model
    sse = np.linalg.norm((y - y_predicted) ** 2)

    print("R-squared (R²) Score for the model:", r_squared)
    print("Sum of Squared Errors (SSE) for the model:", sse)

    return final_lasso_reg

# Now you have two subsets: X_subset_0, y_subset_0, and X_subset_1, y_subset_1
# You can use these subsets to train two different models based on two different datasets.


sample_counts_to_leave_out = 1

# Perform LassoCV with LeavePOut for the first subset
print("Values for Subset 0 with 1 sample left out:")
# Create y_scaler outside the loop
y_scaler = StandardScaler()
model_0 = perform_lasso_cv(X_subset_0, y_subset_0,
                           sample_counts_to_leave_out, y_scaler)

print("-------------------------------------------------------------")

# Perform LassoCV with LeavePOut for the second subset
print("Values for Subset 1 with 1 sample left out:")
model_1 = perform_lasso_cv(X_subset_1, y_subset_1,
                           sample_counts_to_leave_out, y_scaler)

# You now have two models, model_0 and model_1, trained on the respective subsets.
