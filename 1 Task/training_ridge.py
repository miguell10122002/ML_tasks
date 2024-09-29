

import numpy as np
from sklearn.model_selection import LeavePOut
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

X = np.load('X_train_regression1.npy')
y = np.load('y_train_regression1.npy')

alphas = np.arange(0.001, 10, 0.001)

lpo = LeavePOut(1)

best_alpha = None
best_mse = float('inf')

for alpha in alphas:
    mse_scores = []

    for train_index, test_index in lpo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

       
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)

       
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    avg_mse = np.mean(mse_scores)

    if avg_mse < best_mse:
        best_mse = avg_mse
        best_alpha = alpha

print(f"Best Alpha: {best_alpha}, Best Avg. MSE: {best_mse}")