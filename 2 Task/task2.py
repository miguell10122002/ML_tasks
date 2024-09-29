import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeavePOut
from sklearn.metrics import r2_score

from scipy.stats import mode

import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load your data
X = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')

# Combine X and y into a single dataset
combined_data = np.column_stack((X, y))

# Number of runs for robustness assessment
num_runs = 100

# Lists to store clustering results
gmm_results = []

# Run GMM multiple times
for _ in range(num_runs):
    # Create a GMM model with 2 clusters
    gmm = GaussianMixture(n_components=2, random_state=None)

    # Fit the model to the data
    gmm.fit(combined_data)

    # Append cluster assignments to results list
    gmm_results.append(gmm.predict(combined_data))

# Calculate and print consistency measures (e.g., percentage of agreement)
gmm_consistency = np.mean(np.std(gmm_results, axis=0))

# Combine results from multiple runs using majority voting
gmm_combined = mode(gmm_results, axis=0).mode[0]

# Print the combined cluster assignments in a descriptive format
print("\nCombined GMM Clusters:")
cluster_0_samples_gmm = np.where(gmm_combined == 0)[0]
cluster_1_samples_gmm = np.where(gmm_combined == 1)[0]
print(
    f"Cluster 0 (GMM): Samples {cluster_0_samples_gmm} ({len(cluster_0_samples_gmm)} samples)")
print(
    f"Cluster 1 (GMM): Samples {cluster_1_samples_gmm} ({len(cluster_1_samples_gmm)} samples)")

# Create two subsets based on the clusters
subset_0 = combined_data[gmm_combined == 0]
subset_1 = combined_data[gmm_combined == 1]

# Separate the subsets into features (X) and target (y)
X_subset_0 = subset_0[:, :-1]  # Exclude the last column which is the target
y_subset_0 = subset_0[:, -1]   # The last column is the target

X_subset_1 = subset_1[:, :-1]
y_subset_1 = subset_1[:, -1]

# Now you have two subsets: X_subset_0, y_subset_0, and X_subset_1, y_subset_1
# You can use these subsets to train two different models based on two different datasets.

# We've empirically noticed that the best alpha is in between these values.
alpha_values = np.arange(0.01, 10, 0.001)

# Performing a Cross Validation in order to deduct which are the best hyperparameters(alpha) and coefficents
lpo = LeavePOut(p=10)
lasso_cv = LassoCV(alphas=alpha_values, cv=lpo)
lasso_cv.fit(X_subset_0, y_subset_0)

# Storing the desired values in order to analyze some statistics
alpha_values = lasso_cv.alphas_
optimal_alpha = lasso_cv.alpha_
mse_cv_values = lasso_cv.mse_path_.mean(axis=1)
best_mse_cv = mse_cv_values[np.where(lasso_cv.alphas_ == optimal_alpha)]

# Printing some cross validation metrics
print("Optimal Alpha:", optimal_alpha)
rounded_best_msse = "%.3f" % best_mse_cv
print("MSE (Cross-Validation):", rounded_best_msse)

# Calculating the final Lasso Model
final_lasso_reg = Lasso(alpha=optimal_alpha)
final_lasso_reg.fit(X_subset_0, y_subset_0)

# Print R-squared (R²) for the model
y_predicted = final_lasso_reg.predict(X_subset_0)
r_squared = r2_score(y_subset_0, y_predicted)
rounded_r2 = "%.3f" % r_squared
print("R-squared (R²) Score:", rounded_r2)

# Print the final SSE
sse = np.linalg.norm((y_subset_0 - y_predicted) ** 2)
print("The SSE is:", sse)


# We've empirically noticed that the best alpha is in between these values.
alpha_values_1 = np.arange(0.01, 10, 0.001)

# Performing a Cross Validation in order to deduct which are the best hyperparameters(alpha) and coefficents
lpo_1 = LeavePOut(p=10)
lasso_cv_1 = LassoCV(alphas=alpha_values_1, cv=lpo_1)
lasso_cv_1.fit(X_subset_1, y_subset_1)

# Storing the desired values in order to analyze some statistics
alpha_values_1 = lasso_cv_1.alphas_
optimal_alpha_1 = lasso_cv_1.alpha_
mse_cv_values_1 = lasso_cv_1.mse_path_.mean(axis=1)
best_mse_cv_1 = mse_cv_values_1[np.where(lasso_cv.alphas_ == optimal_alpha_1)]

# Printing some cross validation metrics
print("Optimal Alpha:", optimal_alpha_1)
rounded_best_msse_1 = "%.3f" % best_mse_cv_1
print("MSE (Cross-Validation):", rounded_best_msse_1)

# Calculating the final Lasso Model
final_lasso_reg_1 = Lasso(alpha=optimal_alpha_1)
final_lasso_reg_1.fit(X_subset_1, y_subset_1)

# Print R-squared (R²) for the model
y_predicted_1 = final_lasso_reg_1.predict(X_subset_1)
r_squared_1 = r2_score(y_subset_1, y_predicted_1)
rounded_r2_1 = "%.3f" % r_squared_1
print("R-squared (R²) Score:", rounded_r2_1)

# Print the final SSE
sse = np.linalg.norm((y_subset_1 - y_predicted_1) ** 2)
print("The SSE is:", sse)
