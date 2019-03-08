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

def make_int(text):
    if(isinstance(text, str)):
        return int(text.strip('" '))

def apply_models(X_train, y_train, X_dev, y_dev):
    apply_logistic_regression(X_train, y_train, X_dev, y_dev)

def apply_logistic_regression(X_train, y_train, X_dev, y_dev):    
    
    sc = StandardScaler()
#    X_train = sc.fit_transform(X_train)
#    X_dev = sc.fit_transform(X_dev)
    classifier = LogisticRegression(random_state = seed, solver='newton-cg', multi_class='multinomial')
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_dev)
    
    cm = confusion_matrix(y_dev, y_pred)
    ac = accuracy_score(y_dev, y_pred)
    print("********************************************************************************")
    print(classification_report(y_dev, y_pred))
    print(cm)
    print(ac)
    print("********************************************************************************")


#train, validation = train_test_split(train_full, test_size = 0.2, random_state = None)
train_full = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv')
train_full = train_full[np.isfinite(train_full['Victim Age'])]
X_full = train_full.iloc[:,[11]].values
y_full = train_full.iloc[:, 8].values

#divide training data into 10 pairs of train and dev
kfn = kf(n_splits=10, random_state=seed, shuffle=False)
kfn.get_n_splits(X_full)



for train_index, dev_index in kfn.split(X_full):
    #print("train:", train_index, "dev:", dev_index)
    X_train, X_dev = X_full[train_index], X_full[dev_index]
    y_train, y_dev = y_full[train_index], y_full[dev_index]
    apply_models(X_train, y_train, X_dev, y_dev)