import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeavePOut
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Loading the data
X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')

column_index = 0, 4, 7
X_train = np.delete(X_train, column_index, axis=1)
X_test = np.delete(X_test, column_index, axis=1)

# Performing a Cross Validation
lpo = LeavePOut(p=1)
linearReg = LinearRegression()
grid_search = GridSearchCV(
    linearReg, cv=lpo, param_grid={}, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Calculating the Mean SSE for CV
msse = -grid_search.cv_results_['mean_test_score']

rounded_best_msse = "%.3f" % msse
print("Mean SSE (Cross-Validation):", rounded_best_msse)

# Calculate R-squared (R²) for the model
linearReg.fit(X_train, y_train)
y_predicted = linearReg.predict(X_train)
r_squared = r2_score(y_train, y_predicted)
rounded_r2 = "%.3f" % r_squared
print("R-squared (R²) Score:", rounded_r2)

# Calculate the final SSE
sse = np.linalg.norm((y_train - y_predicted) ** 2)
print("The SSE is:", sse)
