from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeavePOut
import numpy as np
import warnings


warnings.simplefilter(action='ignore', category=UserWarning)


X = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')


combined_data = np.column_stack((X, y))


gmm = GaussianMixture(n_components=2, random_state=None)


gmm.fit(combined_data)


gmm_labels = gmm.predict(combined_data)


print("\nGMM Clusters:")
cluster_0_samples_gmm = np.where(gmm_labels == 0)[0]
cluster_1_samples_gmm = np.where(gmm_labels == 1)[0]
print(
    f"Cluster 0 (GMM): Samples {cluster_0_samples_gmm} ({len(cluster_0_samples_gmm)} samples)")
print(
    f"Cluster 1 (GMM): Samples {cluster_1_samples_gmm} ({len(cluster_1_samples_gmm)} samples)\n")


subset_0 = combined_data[gmm_labels == 0]
subset_1 = combined_data[gmm_labels == 1]


X_subset_0 = subset_0[:, :-1]  
y_subset_0 = subset_0[:, -1]   

X_subset_1 = subset_1[:, :-1]
y_subset_1 = subset_1[:, -1]




def perform_ridge_cv(X, y, sample_counts_to_leave_out):
    alpha_values = np.arange(0.23, 0.24, 0.00001)
    num_samples = X.shape[0]

    
    lpo = LeavePOut(p=sample_counts_to_leave_out)

    
    mse_values = np.zeros(len(alpha_values))

  
    for i, alpha in enumerate(alpha_values):
        mse_for_alpha = 0

        for train_index, test_index in lpo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            ridge_reg = Ridge(alpha=alpha)
            ridge_reg.fit(X_train, y_train)

            y_pred = ridge_reg.predict(X_test)
            mse_for_alpha += mean_squared_error(y_test, y_pred)

        mse_values[i] = mse_for_alpha / num_samples

   
    optimal_alpha_index = np.argmin(mse_values)
    optimal_alpha = alpha_values[optimal_alpha_index]

    print("Optimal Alpha:", optimal_alpha)

    
    best_mse = mse_values[optimal_alpha_index]
    print("Mean MSE (Cross-Validation):", best_mse)

    final_ridge_reg = Ridge(alpha=optimal_alpha)
    final_ridge_reg.fit(X, y)

   
    y_predicted = final_ridge_reg.predict(X)
    r_squared = r2_score(y, y_predicted)
    rounded_r2 = "%.3f" % r_squared
    print("R-squared (R²) Score for the model:", rounded_r2)

    
    sse = np.linalg.norm((y - y_predicted) ** 2)
    print("The SSE is:", sse)

    return final_ridge_reg




sample_counts_to_leave_out = 1


print("Values for Subset 0 with 1 sample left out:")
ridge_model_1 = perform_ridge_cv(
    X_subset_1, y_subset_1, sample_counts_to_leave_out)

print("---------------------------")



print("Linear Regression with Leave-One-Out Cross-Validation for Subset 1:")


linear_model_subset_0 = LinearRegression()


loo = LeaveOneOut()


mse_scores_subset_0 = []


for train_index, test_index in loo.split(X_subset_0):
    X_train, X_test = X_subset_0[train_index], X_subset_0[test_index]
    y_train, y_test = y_subset_0[train_index], y_subset_0[test_index]

   
    linear_model_subset_0.fit(X_train, y_train)

   
    y_pred = linear_model_subset_0.predict(X_test)

    
    mse_fold = mean_squared_error(y_test, y_pred)
    mse_scores_subset_0.append(mse_fold)


linear_model_subset_0.fit(X_subset_0, y_subset_0)

mean_mse_subset_0 = np.mean(mse_scores_subset_0)
rounded_mean_mse_subset_0 = "%.3f" % mean_mse_subset_0
print("Mean Squared Error (MSE) for Subset 0:",
      rounded_mean_mse_subset_0)


y_pred_subset_0 = linear_model_subset_0.predict(X_subset_0)
r_squared = r2_score(y_subset_0, y_pred_subset_0)
rounded_r2 = "%.3f" % r_squared
print("R-squared (R²) Score for the model:", rounded_r2)

sse_subset_0 = np.linalg.norm((y_subset_0 - y_pred_subset_0) ** 2)
print("The SSE for Subset 1 (Linear Regression) is:", sse_subset_0)
