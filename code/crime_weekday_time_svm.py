# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:15:59 2018

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
        date = date.month
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

def apply_models(X_train, y_train, X_dev, y_dev):
    #apply_logistic_regression(X_train, y_train, X_dev, y_dev)
    ##apply_decission_tree(X_train, y_train, X_dev, y_dev)
    apply_svm(X_train, y_train, X_dev, y_dev)
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


    
def apply_svm(X_train, y_train, X_dev, y_dev):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_dev = sc.fit_transform(X_dev)
    print("Completed scaling..........")
    
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_dev)

    cm = confusion_matrix(y_dev, y_pred)
    ac = accuracy_score(y_dev, y_pred)
    print(classification_report(y_dev, y_pred))

    dc = classifier.dual_coef_
    sv = classifier.support_vectors_
    
    print(cm)
    print(ac)
    print(dc)
    print(sv)


def main():
    
    #train, validation = train_test_split(train_full, test_size = 0.2, random_state = None)
    train_full = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv', 
                             converters={'Date Occurred': date_time_converter, 
                                         'Time Occurred': format_time, 'Crime Code': format_crime_code})
    print("Completed reading csv.....")
    train_full = train_full[np.isfinite(train_full['Date Occurred'])]
    #train_full = train_full[~np.isnan(train_full['Time Occurred'])]
    train_full = train_full[~np.isnan(train_full['Crime Code'])]
    print(len(train_full))
    X_full = train_full.iloc[:,[3,4,8]].values    
    y_full = train_full.iloc[:, 5].values
    
    #divide training data into 10 pairs of train and dev
    kfn = kf(n_splits=10, random_state=seed, shuffle=False)
    kfn.get_n_splits(X_full)
    i=0;
    for train_index, dev_index in kfn.split(X_full):
        #print("train:", train_index, "dev:", dev_index)
        i +=1
        print("In iteration: ")
        print(i)
        X_train, X_dev = X_full[train_index], X_full[dev_index]
        y_train, y_dev = y_full[train_index], y_full[dev_index]
        apply_models(X_train, y_train, X_dev, y_dev)
    #print_performance()
        
main()