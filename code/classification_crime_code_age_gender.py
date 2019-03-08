# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 22:02:01 2018

@author: mansi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import SGDClassifier

def crime_code(str1):
    return int(str1[0:1])

def age (str1):
    if str1!='':
        return ((int(str1)//5))*5

def weapon(str1):
    if str1=='':
        pass
    else:
        return int(str1[0:1])
    
dataset = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train_type1.csv', 
                      converters = {'Crime Code': crime_code,'Victim Age':age})
dataset = dataset[np.isfinite(dataset['Victim Age'])]
dataset = dataset[np.isfinite(dataset['Weapon Used Code'])]

dataset['Victim Sex'] = dataset["Victim Sex"].replace(['H','-','X'],'U')
#dataset['Victim Sex'].value_counts()
dataset = dataset[dataset['Victim Descent'].notnull()]
dataset=dataset[dataset['Victim Sex'].notnull()]
#dataset.shape

train = pd.concat([dataset.get(['Crime Code']),
                           pd.get_dummies(dataset['Victim Sex'], prefix='Gender'),
                           pd.get_dummies(dataset['Victim Age'], prefix='Age'),
                           pd.get_dummies(dataset['Type 1'],prefix='Type')],axis=1)

X = train.iloc[:,1:22].values
y = train.iloc[:, [22,23,24]].values



dataset1 = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/test_type.csv', 
                       converters = {'Crime Code': crime_code,'Victim Age':age})
dataset1 = dataset1[np.isfinite(dataset1['Victim Age'])]
#dataset1 = dataset1[np.isfinite(dataset1+['Weapon Used Code'])]

dataset1['Victim Sex'] = dataset1["Victim Sex"].replace(['H','-','X'],'U')
#dataset1['Victim Sex'].value_counts()
dataset1 = dataset1[dataset1['Victim Descent'].notnull()]
dataset1=dataset1[dataset1['Victim Sex'].notnull()]
#dataset1.shape

train1 = pd.concat([dataset1.get(['Crime Code']),
                           pd.get_dummies(dataset1['Victim Sex'], prefix='Gender'),
                           pd.get_dummies(dataset1['Victim Age'], prefix='Age'),
                           pd.get_dummies(dataset1['Type 1'],prefix='Type')],axis=1)

X1 = train1.iloc[:,1:22].values
y1 = train1.iloc[:, [22,23,24]].values

#handling the categorical values in victim gender
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:,1:]


X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.25, random_state = 5)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_dev = sc.transform(X_dev)
X1 = sc.transform(X1)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)

#Fitting Logistic Regression to the Training set

#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)

y_pred_train = classifier.predict(X_train)
y_pred_dev = classifier.predict(X_dev)
y_pred_test = classifier.predict(X1) 

#cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_train, y_pred_train)*100
ac1 = accuracy_score(y_dev, y_pred_dev)*100
ac2 = accuracy_score(y1, y_pred_test)*100

print("********************************accuracies******************************")
print("Train: " + str(ac))
print("Validation: " + str(ac1))
print("Test: " + str(ac2))
print("************************************************************************")
