

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
    if(isinstance(value, str)):
        date = pd.to_datetime(value, format='%m/%d/%Y')
        return date

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
def format_crime_code(str1):
    return int(str1[0:1])
def make_int(text):
        if(isinstance(text, str)):
            return int(text.strip('" '))
df = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train_type1.csv',
                 converters={'Date Occurred': date_time_converter, 
                                            'Time Occurred': format_time,
                                            'Crime Code': format_crime_code}) 
columns = df.columns
b=df[['Date Occurred','Time Occurred','Crime Code', 'Area ID']]
a=type(columns)
column_1 = df.ix[:,3]
f=type(column_1)
Y=b['Time Occurred'].values
print(Y)
b['weekday'] = column_1.dt.dayofweek
print(b)
X = b.iloc[:,[2,3, 4]].values   
print(X)
Y = b.iloc[:, 1].values
print(Y)
df1 = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/test_type.csv',converters={'Date Occurred': date_time_converter, 
                                            'Time Occurred': format_time,
                                            'Crime Code': format_crime_code})
columns = df1.columns
bo=df1[['Date Occurred','Time Occurred','Crime Code', 'Area ID']]
ao=type(columns)
column_1o = df1.ix[:,3]
fo=type(column_1)
Yo=b['Time Occurred'].values
bo['weekday'] = column_1o.dt.dayofweek
X_test = bo.iloc[:,[2,3, 4]].values   
y_test = bo.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_dev = sc.fit_transform(X_dev)
X_test=sc.fit_transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred_train = classifier.predict(X_train)
y_pred_dev = classifier.predict(X_dev)
y_pred_test=classifier.predict(X_test)

ac_train = accuracy_score(y_train, y_pred_train)*100
ac_dev = accuracy_score(y_dev, y_pred_dev)*100
ac_test = accuracy_score(y_test, y_pred_test)*100


print("*************************** DT Accuracy:***********************************")
print("Train: " + str(ac_train))
print("Validation: " + str(ac_dev))
print("Test: " + str(ac_test))
print("***************************************************************************")


