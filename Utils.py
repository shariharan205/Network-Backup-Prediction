import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

#Read the data
data = pd.read_csv("network_backup_dataset.csv")


"""==========================================Util methods==========================================================="""

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def scalar_encode_day(days):
    day_to_num = dict(zip(list(calendar.day_name), range(1, 8)))
    return [day_to_num[day] for day in days]

def scalar_encode_underscored(file_names):
    return [int(name.split('_')[-1]) for name in file_names]

def mse(predicted, actual):
    return mean_squared_error(predicted, actual)


def linear_regression(X, y, plot=True, model=LinearRegression()):
    kf = KFold(n_splits=10, shuffle=True)

    cv_train_rmse, cv_test_rmse = [], []
    min_test_rmse = float("inf")

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lm = model
        lm.fit(X_train, y_train)
        train_rmse = mse(lm.predict(X_train), y_train)
        test_rmse = mse(lm.predict(X_test), y_test)

        if test_rmse < min_test_rmse:
            min_test_rmse = test_rmse
            best_model = lm

        cv_train_rmse.append(train_rmse)
        cv_test_rmse.append(test_rmse)

    avg_train_rmse = np.sqrt(np.mean(cv_train_rmse))
    avg_test_rmse = np.sqrt(np.mean(cv_test_rmse))

    print("Coefficients are : ", best_model.coef_)
    best_model_y_predict = best_model.predict(X)

    if plot:

        best_model_y_predict = best_model_y_predict.reshape(y.shape)

        plt.figure(figsize=(8, 5))
        plt.scatter(y, best_model_y_predict, s=5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title("Fitted values vs True values")
        plt.show()