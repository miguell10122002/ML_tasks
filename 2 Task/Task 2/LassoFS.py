from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeavePOut
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Loading the data into vectors
X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')

# column_index = 0, 4, 8, 9
# X_train = np.delete(X_train, column_index, axis=1)

if y_train.ndim == 2:
    y_train = y_train.ravel()

# Calculate the correlation matrix
correlation_matrix = np.corrcoef(X_train, rowvar=False)

print("Correlation Coefficients:")
for i in range(correlation_matrix.shape[0]):
    for j in range(i + 1, correlation_matrix.shape[1]):
        corr = correlation_matrix[i, j]
        feature_i = f'Feature_{i}'
        feature_j = f'Feature_{j}'
        print(f"{feature_i} and {feature_j}: {corr:.2f}")
# Calculate VIF for each feature


def calculate_vif(data_frame):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_df.columns
    vif_data["VIF"] = [1 / (1 - data_frame.iloc[:, i].corr(data_frame.iloc[:, j]) ** 2)
                       for i, j in zip(range(data_frame.shape[1]), range(data_frame.shape[1]))]
    return vif_data


# Create a DataFrame with your feature matrix
feature_df = pd.DataFrame(
    X_train, columns=[f'Feature_{i}' for i in range(X_train.shape[1])])

# Calculate VIF
vif_data = calculate_vif(feature_df)

# Find features with high VIF
vif_threshold = 2  # Adjust the VIF threshold as needed

print("\nVIF Values:")
for index, row in vif_data.iterrows():
    feature = row["Feature"]
    vif_value = row["VIF"]
    if vif_value > vif_threshold:
        print(f"{feature}: {vif_value:.2f}")

# alpha values to be tested [0.001; 3] with steps of 0.001
alpha_values = np.arange(0.001, 3.001, 0.001)

# Performing a Cross Validation in order to deduct which are the best hyperparameters and coefficents
lpo = LeavePOut(p=1)
lasso_cv = LassoCV(alphas=alpha_values, cv=lpo)
lasso_cv.fit(X_train, y_train)

# Storing the desired values in order to plot the graphs of interest
alpha_values = lasso_cv.alphas_
optimal_alpha = lasso_cv.alpha_
msse_values = lasso_cv.mse_path_.mean(axis=1)
best_msse = msse_values[np.where(lasso_cv.alphas_ == optimal_alpha)]

# Plot Mean SSE vs. Lambda
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, msse_values, marker='o', linestyle='-', color='b')
plt.xlabel('Lamba')
plt.ylabel('Mean Sum of Squared Errors (MSSE)')
plt.title('Mean SSE vs. Alpha for Ridge Regression')
plt.grid(True)
plt.show()

print("Optimal Alpha:", optimal_alpha)
rounded_best_msse = "%.3f" % best_msse
print("Mean SSE (Cross-Validation):", rounded_best_msse)

final_lasso_reg = Lasso(alpha=optimal_alpha)
final_lasso_reg.fit(X_train, y_train)
# y_pred = final_lasso_reg.predict(X_test)
