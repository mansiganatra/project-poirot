# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xlwt as xl
from sklearn.model_selection import KFold as kf
#import matplotlib.pyplot as plt
#import seaborn as sns
seed = 5;

dataset = pd.read_csv('D:/Sem1/INF552/Project/Data/cd_kaggle.csv')

train_full, test = train_test_split(dataset, test_size = 0.2, random_state = None)
test_pd = pd.DataFrame(test)
test_pd.to_excel("D:/Sem1/INF552/Project/Data/Final1/test.xlsx")


train_pd = pd.DataFrame(train_full)
train_pd.to_excel("D:/Sem1/INF552/Project/Data/Final1/train.xlsx")