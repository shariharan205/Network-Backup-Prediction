#Importing the necessary libraries.
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
                      
#K Fold Cross validation as a function
def kfold_cv(X,y,return_errors=False,regression='Linear',features=5,trees=20,depth=4,k=5,
             return_model=False):
    train_error_k, test_error_k = [], []
    kf = KFold(n_splits=5,shuffle=False)
    best_mse=1e8
    for trainset, testset in kf.split(X):
        if regression == 'KNN':
            #print('Using KNN Regression')
            regressor = KNeighborsRegressor(n_neighbors=k)
        elif regression == 'RF':
            #print('Using random forest regression')
            regressor = RandomForestRegressor(n_estimators=trees,max_depth=depth,
                                              max_features=features,
                                              bootstrap=True)
        else:
            regressor = LinearRegression()
        regressor.fit(X=X.iloc[trainset],y=y.iloc[trainset].values.ravel())
        y_pred_train = regressor.predict(X.iloc[trainset])
        train_mse = mean_squared_error(y.iloc[trainset],y_pred_train) 
        train_error_k.append(train_mse)
        y_pred_test = regressor.predict(X.iloc[testset])
        test_mse = mean_squared_error(y.iloc[testset],y_pred_test)
        test_error_k.append(test_mse)
        if test_mse < best_mse:
            best_model = regressor
    training_error = np.sqrt(np.mean(train_error_k))
    test_error = np.sqrt(np.mean(test_error_k))    
    print('The training error is', training_error)    
    print('The testing error is', test_error)    
    if return_errors:
        return training_error,test_error
    if return_model:
        return best_model
    
def show_plots(X,y,regressor):
    y = y.values.ravel()
    regressor.fit(X=X,y=y) 
    y_pred = regressor.predict(X)
    plt.scatter(range(len(X)),y_pred,c='red',label='Fitted value',edgecolors='none',zorder=2)
    plt.scatter(range(len(X)),y,c='blue',label='True value',edgecolors='none',zorder=1)
    plt.title('Fitted value vs true value for the model')
    plt.legend()
    plt.show()
    
    plt.scatter(range(len(X)),y_pred,c='red',label='Fitted value',edgecolors='none',zorder=1)
    plt.scatter(range(len(X)),y-y_pred,c='blue',label='Residual',edgecolors='none',zorder=2)
    plt.title('Residuals vs fitted values for the model')
    plt.legend()
    plt.show()  
	
kfold_cv(scalar_data_X,scalar_data_y)
show_plots(scalar_data_X,scalar_data_y,regressor=LinearRegression())
    
###############################################################################
# Scale the data
###############################################################################

sc = StandardScaler()
scaled_X = pd.DataFrame(sc.fit_transform(scalar_data_X))
kfold_cv(scaled_X,scalar_data_y)
show_plots(scaled_X,scalar_data_y,regressor=LinearRegression())

###############################################################################
#  F Regression and Mutual Info Regression
###############################################################################

F,_ = f_regression(scalar_data_X,scalar_data_y.values.ravel())
top_features = np.argsort(F)[-3:]
print('The top features are', scalar_data_X.columns.values[top_features])
f_top_X = scalar_data_X[scalar_data_X.columns.values[top_features]]
kfold_cv(f_top_X,scalar_data_y)
show_plots(f_top_X,scalar_data_y,regressor=LinearRegression())

mi = mutual_info_regression(scalar_data_X,scalar_data_y.values.ravel())
top_features = np.argsort(mi)[-3:]
print('The top features are', scalar_data_X.columns.values[top_features])
f_top_X = scalar_data_X[scalar_data_X.columns.values[top_features]]
kfold_cv(f_top_X,scalar_data_y)
show_plots(f_top_X,scalar_data_y,regressor=LinearRegression())

###############################################################################
# One-Hot encoding
###############################################################################

onehotencoder = OneHotEncoder(categorical_features = 'all')
onehot_week = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['Week #'].reshape(-1,1)).toarray()[:,1:])
onehot_day = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['day'].reshape(-1,1)).toarray()[:,1:])
onehot_time = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['Backup Start Time - Hour of Day'].reshape(-1,1)).toarray()[:,1:])
onehot_workflow = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['Workflow_ID'].reshape(-1,1)).toarray()[:,1:])
onehot_file = pd.DataFrame(onehotencoder.fit_transform(scalar_data_X['File_number'].reshape(-1,1)).toarray()[:,1:])

scalar_week = pd.DataFrame(scalar_data_X['Week #'])
scalar_day = pd.DataFrame(scalar_data_X['day'])
scalar_time = pd.DataFrame(scalar_data_X['Backup Start Time - Hour of Day'])
scalar_workflow = pd.DataFrame(scalar_data_X['Workflow_ID'])
scalar_file = pd.DataFrame(scalar_data_X['File_number'])

d = {0:[scalar_week,onehot_week],1:[scalar_day,onehot_day],2:[scalar_time,onehot_time],
        3:[scalar_workflow,onehot_workflow],4:[scalar_file,onehot_file]}

train_err,test_err = [], []
for number in range(32):
    num = np.binary_repr(number, width=5)
    frames = []
    for i in range(len(num)):
        if int(num[i])==0:
            frames.append(d[i][0])
        else:
            frames.append(d[i][1])            
    result = pd.concat(frames,axis=1)
    train,test = kfold_cv(result,scalar_data_y,return_errors=True)
    train_err.append(train)
    test_err.append(test)
    
plt.plot(range(1,33),test_err,c='blue',label='Testing error')
plt.legend()
plt.xlabel('Combination number')
plt.ylabel('Root mean square error')
plt.title('Testing error for different combinations')
plt.show()    

plt.plot(range(1,33),train_err,c='red',label='Training error')
plt.legend()
plt.xlabel('Combination number')
plt.ylabel('Root mean square error')
plt.title('Training error for different combinations')
plt.show()    

###############################################################################
# Accuracy vs Degree of Polynomial in Regression
###############################################################################

num_workflow = len(set(scalar_data['Workflow_ID']))
for i in range(num_workflow):
    subset_data = scalar_data[scalar_data['Workflow_ID']==i]
    y = subset_data[[-1]]
    X = subset_data[['Week #', 'day', 'Backup Start Time - Hour of Day','File_number']]
    print('The workflow ID number is', i)
    best_model = kfold_cv(X,y,return_model=True)
    show_plots(X,y,best_model)
    
    
for i in range(num_workflow):
    train_err = []
    test_err = []
    subset_data = scalar_data[scalar_data['Workflow_ID']==i]
    y = subset_data[[-1]]
    X = subset_data[['Week #', 'day', 'Backup Start Time - Hour of Day','File_number']]
    for degree in range(2,10):
        poly_reg = PolynomialFeatures(degree = degree)
        X_poly = pd.DataFrame(poly_reg.fit_transform(X))
        train,test=kfold_cv(X_poly,y,return_errors=True)
        train_err.append(train)
        test_err.append(test)
        
    plt.plot(range(2,10),train_err,c='red',label='Training error')
    plt.plot(range(2,10),test_err,c='blue',label='Testing error')
    plt.legend()
    plt.xlabel('Polynomial degree')
    plt.ylabel('Root mean square error')
    plt.title('Error for polynomial degrees for workflow id = %d'%i)
    plt.show()    