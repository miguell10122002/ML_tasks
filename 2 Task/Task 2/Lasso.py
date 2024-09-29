import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeavePOut
from sklearn.metrics import r2_score
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

# Define a function for LassoCV with LeavePOut


def perform_lasso_cv(X, y, sample_counts_to_leave_out):
    alpha_values = np.arange(0, 50, 0.1)

    # Create the LeavePOut cross-validator
    lpo = LeavePOut(p=sample_counts_to_leave_out)

    # Perform LassoCV with LeavePOut
    lasso_cv = LassoCV(alphas=alpha_values, cv=lpo)
    lasso_cv.fit(X, y)

    # Storing the desired values for plotting
    alpha_values = lasso_cv.alphas_
    optimal_alpha = lasso_cv.alpha_
    msse_values = lasso_cv.mse_path_.mean(axis=1)
    best_msse = msse_values[np.where(lasso_cv.alphas_ == optimal_alpha)]

    print(f"Optimal Alpha:", optimal_alpha)
    rounded_best_msse = "%.3f" % best_msse
    print("Mean SSE (Cross-Validation):", rounded_best_msse)

    final_lasso_reg = Lasso(alpha=optimal_alpha)
    final_lasso_reg.fit(X, y)

    # Print R-squared (R²) for the model
    y_predicted = final_lasso_reg.predict(X)
    r_squared = r2_score(y, y_predicted)
    rounded_r2 = "%.3f" % r_squared
    print("R-squared (R²) Score for the model:", rounded_r2)

    # Print the final SSE
    sse = np.linalg.norm((y - y_predicted) ** 2)
    print("The SSE is:", sse)

    return final_lasso_reg

# Now you have two subsets: X_subset_0, y_subset_0, and X_subset_1, y_subset_1
# You can use these subsets to train two different models based on two different datasets.


sample_counts_to_leave_out = 1

# Perform LassoCV with LeavePOut for the first subset
print("Values for Subset 0 with 1 sample left out:")
model_0 = perform_lasso_cv(X_subset_0, y_subset_0, sample_counts_to_leave_out)

print("-------------------------------------------------------------")

# Perform LassoCV with LeavePOut for the second subset
print("Values for Subset 1 with 1 sample left out:")
model_1 = perform_lasso_cv(X_subset_1, y_subset_1, sample_counts_to_leave_out)

# You now have two models, model_0 and model_1, trained on the respective subsets.
