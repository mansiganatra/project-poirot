# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:18:42 2018

@author: mansi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xlwt as xl
from sklearn.model_selection import KFold as kf
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#import seaborn as sns

#import matplotlib.pyplot as plt
#import seaborn as sns
seed = 5;
def date_time_converter(value):
    #if(isinstance(value, str)):
        date = pd.to_datetime(value, format='%m/%d/%Y')
        return date

def format_time(time):
    new_time = time
    if(len(time)==1):
        new_time = "0" +time + "00"
    elif(len(time)==2 and int(time) <=23):
        new_time = time + "00"
    elif(len(time)==2 and int(time) > 23):
        new_time = "00" + time
    elif(len(time)==3):
        new_time = "0" + time
    return new_time

def make_int(text):
        if(isinstance(text, str)):
            return int(text.strip('" '))

def apply_models(X_train, y_train, X_dev, y_dev):
    apply_linear_regression(X_train, y_train, X_dev, y_dev)

def apply_linear_regression(X_train, y_train, X_dev, y_dev):    
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
#    b1 = regressor.coef_
#    b0 = regressor.intercept_
    y_pred = regressor.predict(X_dev)
#    m=y_pred.astype(int)
    #print(X_test)
    ##c=int(y_pred)
    #print(m)
    print(y_pred)
    mse = mean_squared_error(y_dev,y_pred)
    rmse = mse**0.5
    print(mse)
    print(rmse)
    
    cm = confusion_matrix(y_dev, y_pred)
    #ac = accuracy_score(y_dev, y_pred)
    print("********************************************************************************")
    print(classification_report(y_dev, y_pred))
    print(cm)
    print(ac)
    print("********************************************************************************")


#train, validation = train_test_split(train_full, test_size = 0.2, random_state = None)
train_full = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv',
                         converters={'Date Occurred': date_time_converter, 'Time Occurred' : format_time})
print("Completed reading csv.....")
date_occurred = train_full['Date Occurred']


#train_full = train_full[np.isfinite(train_full['Date Occurred'])]
#train_full = train_full[np.isfinite(train_full['Time Occurred'])]
#train_full = train_full[~np.isnan(train_full['Crime Code'])]
print(len(train_full))
#train_full['Date Occurred'].replace(' ', np.nan, inplace =  True)
#train_full.dropna(subset=['Date Occurred'], inplace=True)
#train_full['Time Occurred'].replace(' ', np.nan, inplace =  True)
#train_full.dropna(subset=['Time Occurred'], inplace=True)
#train_full['Crime Code'].replace(' ', np.nan, inplace =  True)
#train_full.dropna(subset=['Crime Code'], inplace=True)
b=train_full[['Date Occurred','Time Occurred','Crime Code']]
column_1 = train_full.iloc[:,3]
b['weekday'] = column_1.dt.dayofweek
#print(X)

X_full = b['weekday'].values
X_full = X_full.reshape(-1,1)

y_full1 = b['Time Occurred'].values
format_time(b)
b.assign(newtime=pd.to_datetime(b['Time Occurred'], format='%H%M').dt.strftime('%H:%M'))
y_full = b['Time Occurred'].values

#divide training data into 10 pairs of train and dev
kfn = kf(n_splits=10, random_state=seed, shuffle=False)
kfn.get_n_splits(X_full)



for train_index, dev_index in kfn.split(X_full):
    #print("train:", train_index, "dev:", dev_index)
    X_train, X_dev = X_full[train_index], X_full[dev_index]
    y_train, y_dev = y_full[train_index], y_full[dev_index]
    apply_models(X_train, y_train, X_dev, y_dev)