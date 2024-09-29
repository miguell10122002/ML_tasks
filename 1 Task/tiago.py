from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeavePOut
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Loading the data into vectors
X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')

column_index = 0, 4, 7, 8
X_train = np.delete(X_train, column_index, axis=1)
X_test = np.delete(X_test, column_index, axis=1)

if y_train.ndim == 2:
    y_train = y_train.ravel()

# alpha values to be tested [0.001; 3] with steps of 0.001
alpha_values = np.arange(0.0001, 0.5, 0.0001)

# Performing a Cross Validation in order to deduct which are the best hyperparameters and coefficents
lpo = LeavePOut(p=1)
lasso_cv = LassoCV(alphas=alpha_values, cv=lpo)
lasso_cv.fit(X_train, y_train)

# Storing the desired values in order to plot the graphs of interest
alpha_values = lasso_cv.alphas_
optimal_alpha = lasso_cv.alpha_
mse_values = lasso_cv.mse_path_.mean(axis=1)
best_mse = mse_values[np.where(lasso_cv.alphas_ == optimal_alpha)]

# Plot Mean SSE vs. Lambda
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, mse_values, marker='o', linestyle='-', color='b')
plt.xlabel('Lamba')
plt.ylabel('Mean of Squared Errors (MSE)')
plt.title('MSE vs. Alpha for Ridge Regression')
plt.grid(True)
plt.show()

print("Optimal Alpha:", optimal_alpha)
rounded_best_msse = "%.3f" % best_mse
print("MSE (Cross-Validation):", rounded_best_msse)

final_lasso_reg = Lasso(alpha=optimal_alpha)
final_lasso_reg.fit(X_train, y_train)

# Calculate R squared for the model
y_predicted = final_lasso_reg.predict(X_train)
r_squared = r2_score(y_train, y_predicted)
rounded_r2 = "%.3f" % r_squared
print("R-squared (R²) Score:", rounded_r2)
y_pred = final_lasso_reg.predict(X_test)
# Calculate the final SSE
sse = np.linalg.norm((y_train - y_predicted) ** 2)
print("The SSE is:", sse)
x_val = final_lasso_reg.coef_
np.save('Y_test_regression1.npy', y_pred)