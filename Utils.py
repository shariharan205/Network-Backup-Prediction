"""
Utils file for Network Backup Prediction 
Consists methods that encodes the data, performs regression, regularization and optimization
"""
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

        plt.figure(figsize=(8, 5))
        plt.scatter(best_model_y_predict, y - best_model_y_predict, s=5)
        plt.xlabel("Fitted values")
        plt.ylabel("Residual values")
        plt.title("Residual values vs Fitted values")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(y)), y, c='g', marker='x', label='True values')
        plt.scatter(range(len(y)), best_model_y_predict, c='b', marker='o', label='Predicted Values')
        plt.title("Fitted values and True values")
        plt.legend(loc='upper left')
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(y)), best_model_y_predict, c='g', s=5, zorder=2, label='Fitted values')
        plt.scatter(range(len(y)), np.subtract(y, best_model_y_predict), c='b', s=5, zorder=1, label='Residual Values')
        plt.title("Fitted values and Residual values")
        plt.legend(loc='upper left')
        plt.yscale('log')
        plt.show()

    return avg_train_rmse, avg_test_rmse


def f_regr(X, Y):
   return f_regression(X,Y,center = False)

def mutual_info_regr(X, Y):
    return mutual_info_regression(X, Y)


def scalar_encode(data, input_features, output_col_name):
    X = data[input_features].values

    y = data[output_col_name].values

    X[:, 1] = scalar_encode_day(X[:, 1])
    X[:, 3] = scalar_encode_underscored(X[:, 3])
    X[:, 4] = scalar_encode_underscored(X[:, 4])

    return X, y


"""===============================Plotting backup sizes for all workflows for a 20day and 105day period====================="""

data['day_number'] = (data['Week #']-1)*7+ scalar_encode_day(data['Day of Week'])
data['Workflow_ID'] = scalar_encode_underscored(data['Work-Flow-ID'])

data_20 = data[data['day_number']<=20]

plt.figure(figsize=(8,5))
plt.scatter(x=data_20['day_number'],y=data_20['Size of Backup (GB)'],c=data_20['Workflow_ID'], s = 8)
plt.xlabel('Day number')
plt.ylabel('Backup size (GB)')
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(x=data['day_number'],y=data['Size of Backup (GB)'],c=data['Workflow_ID'], s = 8)
plt.xlabel('Day number')
plt.ylabel('Backup size (GB)')
plt.show()
