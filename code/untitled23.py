# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:21:06 2018

@author: mansi
"""


import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold as kf
from sklearn.model_selection import train_test_split

def date_time_converter(value):
    if(isinstance(value, str)):
        date = pd.to_datetime(value, format='%m/%d/%Y')
        return date
def format_crime_code(str1):
    return int(str1[0:1])
def format_time(m):
   # print(m)
    new_time='0'
    time=m
    if(len(time)==1):
        new_time="0"+time
    elif(len(time)==2 and int(time)<24):
        new_time=time+"00"
    elif(len(time)==2 and int(time)>=24):
        new_time="00"+time
    elif(len(time)==3):
        new_time="0"+time
    elif(len(time)==4):
        new_time=time
    final_time=new_time[0:2]
    return final_time
    print(final_time)


df=pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv',
               converters={'Date Occurred': date_time_converter,'Time Occurred':format_time})   
columns = df.columns
b=df[['Date Occurred','Time Occurred','Crime Code']]

a=type(columns)

column_1 = df.ix[:,3]
f=type(column_1)

b['weekday'] = column_1.dt.dayofweek
b['month'] = column_1.dt.month

X = b.iloc[:,[2,3,4]].values

#Y=b['Time Occurred'].values
Y = b.iloc[:,[1]].values
df1 = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/test_type.csv',converters={'Date Occurred': date_time_converter, 
                                            'Time Occurred': format_time,
                                            'Crime Code': format_crime_code})
columns = df.columns
b1=df[['Date Occurred','Time Occurred','Crime Code', 'Area ID']]
a=type(columns)
column_1 = df.ix[:,3]
f=type(column_1)
b1['weekday'] = column_1.dt.dayofweek
b1['month'] = column_1.dt.month

X_test = b.iloc[:,[2,3,4]].values

#Y=b['Time Occurred'].values
y_test= b.iloc[:,[1]].values


model = Sequential()
model.add(Dense(150, input_dim=3  , activation='relu'))
###model.add(Dense(150, input_dim=3, activation='relu'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(24, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
accuracies = []
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.25, random_state=0)
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
# evaluate the model


scores_train = model.evaluate(X_train, y_train, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
accuracies.append(scores_train[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))


scores_dev = model.evaluate(X_dev, y_dev, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores_dev[1]*100))
accuracies.append(scores_dev[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))


scores_test = model.evaluate(X_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))
accuracies.append(scores_test[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))


print("********************************accuracies******************************")
print("Train: " + str(accuracies[0]))
print("Validation: " + str(accuracies[1]))
print("Test: " + str(accuracies[2]))
print("************************************************************************")
