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
