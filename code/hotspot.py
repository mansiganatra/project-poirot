# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:18:42 2018

@author: mansi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:33:47 2018

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
#import seaborn as sns

#import matplotlib.pyplot as plt
#import seaborn as sns
seed = 5;
def date_time_converter(value):
    #if(isinstance(value, str)):
        date = pd.to_datetime(value, format='%m/%d/%Y')
        date = date.toordinal()
        return date
def format_time(b):
    new_b = []
    new_time ='';
    for time in b:
        if(len(time)==1):
            new_time = "0" +time + "00"
            print(new_time)
        elif(len(time)==2):
            new_time = time + "00"
        elif(len(time)==3):
            new_time = "0" + time
        elif(len(time) == 4):
            new_time = time
    return int(new_time[0:2])

def make_int(text):
        if(isinstance(text, str)):
            return int(text.strip('" '))

def apply_models(X_train, y_train, X_dev, y_dev):
    apply_logistic_regression(X_train, y_train, X_dev, y_dev)

def apply_logistic_regression(X_train, y_train, X_dev, y_dev):    
    
    sc = StandardScaler()
#    X_train = sc.fit_transform(X_train)
#    X_dev = sc.fit_transform(X_dev)
    
    print(X_train[1])
    classifier = LogisticRegression(random_state = seed)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_dev)
    
    cm = confusion_matrix(y_dev, y_pred)
    ac = accuracy_score(y_dev, y_pred)
    print("********************************************************************************")
    print(classification_report(y_dev, y_pred))
    print(cm)
    print(ac)
    print("********************************************************************************")

#def apply_svm()
#train, validation = train_test_split(train_full, test_size = 0.2, random_state = None)
train_full = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv', 
                         converters={'Date Occurred': date_time_converter, 'Time Occurred': format_time})
print("Completed reading csv.....")
date_occurred = train_full['Date Occurred']
print("non float in date: ..................................")
for y in date_occurred:
    if not (isinstance(y,(float,int, long))):
        print("value:")
        print(y)


train_full = train_full[np.isfinite(train_full['Date Occurred'])]
train_full = train_full[np.isfinite(train_full['Time Occurred'])]
train_full = train_full[~np.isnan(train_full['Crime Code'])]
print(len(train_full))
#train_full['Date Occurred'].replace(' ', np.nan, inplace =  True)
#train_full.dropna(subset=['Date Occurred'], inplace=True)
#train_full['Time Occurred'].replace(' ', np.nan, inplace =  True)
#train_full.dropna(subset=['Time Occurred'], inplace=True)
#train_full['Crime Code'].replace(' ', np.nan, inplace =  True)
#train_full.dropna(subset=['Crime Code'], inplace=True)

X_full = train_full.iloc[:,[3,4,8]].values
print("**********before********************")
print(X_full.dtype.names)
print(X_full[1])

y_full = train_full.iloc[:, 26].values


#divide training data into 10 pairs of train and dev
kfn = kf(n_splits=10, random_state=seed, shuffle=False)
kfn.get_n_splits(X_full)



for train_index, dev_index in kfn.split(X_full):
    #print("train:", train_index, "dev:", dev_index)
    X_train, X_dev = X_full[train_index], X_full[dev_index]
    y_train, y_dev = y_full[train_index], y_full[dev_index]
    apply_models(X_train, y_train, X_dev, y_dev)