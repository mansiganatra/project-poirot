# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:22:26 2018

@author: mansi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
#import seaborn as sns

#importing the data set
dataset = pd.read_excel('D:/Sem1/INF552/Project/Data/Final/train.xlsx')
X = dataset.iloc[:,[7,10]]
y = dataset.iloc[:, 1].values

#filling age null values with mean
X=X.fillna(X.mean()['Victim Age']).values

#importing the validation dataset

#importing data test of test
dataset1 = pd.read_excel('D:/Sem1/INF552/Project/Data/Final/validation.xlsx')
X1 = dataset1.iloc[:,[7,10]]
y1 = dataset1.iloc[:, 1].values

sc = StandardScaler()
X = sc.fit_transform(X)
X1 = sc.transform(X1)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)