# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 21:55:47 2018

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
from sklearn.svm import SVC
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
        date = date.toordinal()
        return date
def format_time(b):
    new_time ='';
    for time in b:
        if(len(time)==1):
            new_time = "0" +time + "00"
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


def main():
    
    #train, validation = train_test_split(train_full, test_size = 0.2, random_state = None)
    train_full = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv', 
                             converters={'Date Occurred': date_time_converter, 'Time Occurred': format_time})
    print("Completed reading csv.....")
    train_full = train_full[np.isfinite(train_full['Date Occurred'])]
    train_full = train_full[np.isfinite(train_full['Time Occurred'])]
    train_full = train_full[~np.isnan(train_full['Crime Code'])]
    print(len(train_full))
    X_full = train_full.iloc[:,[3,4,8]].values    
    y_full = train_full.iloc[:, 5].values
    
    
#    #divide training data into 10 pairs of train and dev
#    kfn = kf(n_splits=10, random_state=seed, shuffle=False)
#    kfn.get_n_splits(X_full)
#    i=0;
#    for train_index, dev_index in kfn.split(X_full):
#        #print("train:", train_index, "dev:", dev_index)
#        i +=1
#        print("In iteration: ")
#        print(i)
#        X_train, X_test = X_full[train_index], X_full[dev_index]
#        y_train, y_test = y_full[train_index], y_full[dev_index]
    
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.20, random_state = 0)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Fitting Kernel SVM to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 350, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    print(ac)
    print(classification_report(y_test, y_pred))
    
    
#    # Applying k-Fold Cross Validation
#    from sklearn.model_selection import cross_val_score
#    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 15)
#    acMean = accuracies.mean()
#    acStd = accuracies.std()
#    
#    print("acMean:")
#    print(acMean)
#    print("acStd:")
#    print(acStd)
    
    # Applying Grid Search to find the best model and the best parameters
    from sklearn.model_selection import GridSearchCV
    parameters = {"n_estimators": [100, 300, 500],
                  "max_depth": [3, 5, 7],
                  "min_samples_split": [15, 20, 25],
                  "min_samples_leaf": [5, 10, 15],
                  "max_leaf_nodes": [10, 20, 30],
                  "min_weight_fraction_leaf": [0.1, 0.05, 0.005]}
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 10,
                               n_jobs = -1)
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    
    print("best_accuracy: ")
    print(best_accuracy)
    
    print("best_parameters: ")
    print(best_parameters)
main()