#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor

#Loading data
data = pd.read_csv('network_backup_dataset.csv')

#Finding the day number
day_map = {'Monday':1, 'Tuesday':2, 'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
day_encoded = [day_map[i] for i in data['Day of Week']]
data['day'] = day_encoded
data['day_number'] = (data['Week #']-1)*7+day_encoded
data['Workflow_ID'] = [int(i[10]) for i in data['Work-Flow-ID']]

data_20 = data[data['day_number']<=20]
plt.scatter(x=data_20['day_number'],y=data_20['Size of Backup (GB)'],c=data_20['Workflow_ID'])
plt.xlabel('Day number')
plt.ylabel('Backup size (GB)')
plt.legend()
plt.show()

plt.scatter(x=data['day_number'],y=data['Size of Backup (GB)'],c=data['Workflow_ID'])
plt.xlabel('Day number')
plt.ylabel('Backup size (GB)')
plt.legend()
plt.show()

#Getting the numerical value as a column 
data['File_number'] = [int(i[5:]) for i in data['File Name']]
scalar_data = data[['Week #','day','Backup Start Time - Hour of Day','Workflow_ID',
                    'File_number','Size of Backup (GB)']]
scalar_data_X = scalar_data[['Week #','day','Backup Start Time - Hour of Day',
'Workflow_ID','File_number']]
scalar_data_y = scalar_data[['Size of Backup (GB)']]                     
