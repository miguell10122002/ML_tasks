import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeavePOut
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Loading the data into vectors
X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')

column_index = 0, 4, 7, 8
X_train = np.delete(X_train, column_index, axis=1)
X_test = np.delete(X_test, column_index, axis=1)

# alpha values to be tested [0.001; 3] with steps of 0.001
alpha_values = np.arange(0.001, 3.001, 0.001)

# Performing a Cross Validation in order to deduct which are the best hyperparameters and coefficents
lpo = LeavePOut(p=1)
param_grid = {'alpha': alpha_values}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=lpo,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Checking the best alpha form the CV
optimal_alpha = grid_search.best_params_['alpha']
print("Optimal Alpha:", optimal_alpha)

# Calculating the mean SSE
best_alpha_index = grid_search.best_index_
best_mse = -grid_search.cv_results_['mean_test_score'][best_alpha_index]
rounded_best_msse = "%.3f" % best_mse
print("MSE (Cross-Validation):", rounded_best_msse)

final_ridge_reg = Ridge(alpha=optimal_alpha)
final_ridge_reg.fit(X_train, y_train)

# Calculate R-squared (R²) for the model
y_predicted = final_ridge_reg.predict(X_train)
r_squared = r2_score(y_train, y_predicted)
rounded_r2 = "%.3f" % r_squared
print("R-squared (R²) Score:", rounded_r2)

# Calculate the final SSE
sse = np.linalg.norm((y_train - y_predicted) ** 2)
print("The SSE is:", sse)
