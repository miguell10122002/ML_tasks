import numpy as np

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeavePOut

X_train = np.load('X_train_regression1.npy')
y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')

alphas = np.arange(0.001, 10, 0.001)

column_index = 0,4,8


X_train = np.delete(X_train, column_index, axis=1)
X_test = np.delete(X_test, column_index, axis=1)

lpo = LeavePOut(p=1)

sse_scores = []

lasso_cv = LassoCV(alphas=alphas, cv=lpo)

lasso_cv.fit(X_train, y_train)

optimal_alpha = lasso_cv.alpha_
print("Optimal Alpha:", optimal_alpha)

final_lasso_reg = Lasso(alpha=optimal_alpha)

for train_index, val_index in lpo.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    final_lasso_reg.fit(X_train_fold, y_train_fold)

    val_predictions = final_lasso_reg.predict(X_val_fold)

    fold_sse = np.sum((y_val_fold - val_predictions) ** 2)
    sse_scores.append(fold_sse)

mean_cross_val_sse = np.mean(sse_scores)
print("Mean SSE (Cross-Validation):", mean_cross_val_sse)


y_pred = final_lasso_reg.predict(X_test)
np.save('y_pred.npy', y_pred)