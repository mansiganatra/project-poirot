# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:36:24 2018

@author: mansi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:32:55 2018

@author: mansi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:47:53 2018

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
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#import seaborn as sns

#import matplotlib.pyplot as plt
#import seaborn as sns
seed = 5;
classification_reports = []
accuracies =[]
confusion_matrices = []
def date_time_converter(value):
    #if(isinstance(value, str)):
        date = pd.to_datetime(value, format='%m/%d/%Y')
        return date
def season_crime(month):
    if month in range(1,4):
        season="spring"
    elif month in range(4,7):
        season="summer"
    elif month in range(7,11):
        season="monsoon"
    elif month in range(11,13):
        season="winter"
    return season
    print(season)
def format_time(b):
    new_time = ''
    time = b
    if not time:
        return ''
    elif(len(time)==1):
        new_time = "0" +time + "00"
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

def crime_code(str1):
    return int(str1[0:1])
def isWeekday(day):
    if day in range(1,5):
        return 0
    else:
        return 1
def weapon(str1):
    if str1=='':
        pass
    else:
        return int(str1[0:1])

df=pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv',
               converters={'Date Occurred': date_time_converter,'Time Occurred':format_time,
                           'Crime Code': crime_code,'Weapon Used Code' : weapon})  
df=df[np.isfinite(df['Weapon Used Code'])]
columns = df.columns
b=df[['Date Occurred','Time Occurred','Crime Code', 'Area ID']]

a=type(columns)

column_1 = df.ix[:,3]
f=type(column_1)

b['weekday'] = column_1.dt.dayofweek
b['month'] = column_1.dt.month
b['isWeekend'] = b['weekday'].apply(isWeekday)
b['season']=b['month'].apply(season_crime)


b = pd.concat([b.get(['Time Occurred','Date Occurred','Crime Code','Area ID','weekday','month','isWeekend','Weapon Used Code']),
#                           pd.get_dummies(b['Time Occurred'], prefix='Hour'),
#                           pd.get_dummies(b['Crime Code'], prefix='Crime'),
#                           pd.get_dummies(b['Area ID'], prefix='Area'),
#                           pd.get_dummies(b['weekday'], prefix='day'),
#                           pd.get_dummies(b['month'], prefix='month'),
#                           pd.get_dummies(b['isWeekend'], prefix = 'isWeekend'),
#                           pd.get_dummies(b['Weapon Used Code'], prefix='Weapon'),
                           pd.get_dummies(b['season'],prefix='season')],axis=1)

X = b.iloc[:,2:].values

#Y=b['Time Occurred'].values
Y = b.iloc[:,0].values


def apply_models(X_train, y_train, X_dev, y_dev):
    #apply_logistic_regression(X_train, y_train, X_dev, y_dev)
    apply_decission_tree(X_train, y_train, X_dev, y_dev)
    #apply_svm(X_train, y_train, X_dev, y_dev)
    #apply_knn(X_train, y_train, X_dev, y_dev)


    
def print_performance():
    print("**************************Accuracies:******************************************************")
    for ac in accuracies:
        print(ac)
    print("**************************Classification:******************************************************")
    for cl in classification_reports:
        print(cl)
    print("**************************Classification:******************************************************")
    for cm in confusion_matrices:
        print(cm)
    print("********************************************************************************")
    with open("D:/Sem1/INF552/Project/Data/Results/accuracies.json", 'w+') as f:
        json.dumps(accuracies, f)
    with open("D:/Sem1/INF552/Project/Results/classification_reports.json", 'w+') as f:
        json.dumps(classification_reports, f)
#    with open("D:/Sem1/INF552/Project/Data/Results/confusion_matrices.json", 'w+') as f:
#        json.dumps(confusion_matrices, f)

    

def apply_knn(X_train, y_train, X_dev, y_dev):
    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    
    #Train the model using the training sets
    knn.fit(X_train, y_train)
    
    #Predict the response for test dataset
    y_pred = knn.predict(X_dev)
    
    cm = confusion_matrix(y_dev, y_pred)
    ac = accuracy_score(y_dev, y_pred)
#    print(cm)
    print(ac)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
apply_knn(X_train, y_train, X_test, y_test)