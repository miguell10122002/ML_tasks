from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeavePOut
from sklearn.metrics import r2_score

import numpy as np

# Loading the data into vectors
X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')

# Results of our feature selection
column_index = 0, 4, 7, 8
X_train = np.delete(X_train, column_index, axis=1)
X_test = np.delete(X_test, column_index, axis=1)

if y_train.ndim == 2:
    y_train = y_train.ravel()

# We've empirically noticed that the best alpha is in between these values.
alpha_values = np.arange(0.01635, 0.01640, 0.00000001)

# Performing a Cross Validation in order to deduct which are the best hyperparameters(alpha) and coefficents
lpo = LeavePOut(p=1)
lasso_cv = LassoCV(alphas=alpha_values, cv=lpo)
lasso_cv.fit(X_train, y_train)

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
final_lasso_reg.fit(X_train, y_train)

# Print R-squared (R²) for the model
y_predicted = final_lasso_reg.predict(X_train)
r_squared = r2_score(y_train, y_predicted)
rounded_r2 = "%.3f" % r_squared
print("R-squared (R²) Score:", rounded_r2)

# Print the final SSE
sse = np.linalg.norm((y_train - y_predicted) ** 2)
print("The SSE is:", sse)

y_prediction = final_lasso_reg.predict(X_test)
y_prediction = np.reshape(y_prediction, (-1, 1))
print("The shape of the final vector is:", y_prediction.shape)
np.save('y_test.npy', y_prediction)
