# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:37:20 2018

@author: mansi
"""

import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt
def format_time(m):
   new_time = '0'
   time = m
   if(len(time)==1):
       new_time = "0" +time
   elif(len(time)==2 and int(time)<24):
       new_time = time + "00"
   elif(len(time)==2 and int(time)>=24):
       new_time = "00" + time
   elif(len(time)==3):
       new_time = "0" + time
   elif(len(time) == 4):
       new_time = time
   final_time = new_time[0:2]
   return final_time
   print(final_time)

def date_time_converter(value):
    if(isinstance(value, str)):
        date = pd.to_datetime(value, format='%m/%d/%Y')
        return date
df=pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv',
               converters={'Date Occurred': date_time_converter,'Time Occured':format_time})   
##df=pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv',converters={'Date Occurred': date_time_converter})
columns = df.columns
b=df[['Date Occurred','Time Occurred','Crime Code']]

##print(columns)
a=type(columns)
##print(a)

##print(b)
column_1 = df.iloc[:,3]
f=type(column_1)
##print(f)
#o=pd.DataFrame({"dayofweek": column_1.dt.dayofweek})

b['weekday'] = column_1.dt.dayofweek
#print(b)
X = b['weekday'].values
#print(X)
Y=b['Time Occurred'].values
#print(Y)
from sklearn.model_selection import train_test_split
X = X.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
##from sklearn.preprocessing import StandardScaler, MinMaxScaler
##sc_X = StandardScaler()
##X_train = sc_X.transform(X_train)
##X_test = sc_X.transform(X_test)
##print(X_test)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
b1 = regressor.coef_
b0 = regressor.intercept_
y_pred = regressor.predict(X_test)
m=y_pred.astype(int)
#print(X_test)
##c=int(y_pred)
#print(m)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)
rmse = mse**0.5
print(rmse)