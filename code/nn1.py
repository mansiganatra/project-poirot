# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 20:53:21 2018

@author: mansi
"""

# -- coding: utf-8 --
"""
Created on Mon Oct 29 20:26:20 2018

@author: ritwi
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import KFold as kf
from sklearn.model_selection import train_test_split
# fix random seed for reproducibility
np.random.seed(7)

def crime_code(str1):
    return int(str1[0:1])

def weapon(str1):
    if str1=='':
        pass
    else:
        return int(str1[0:1])
    
dataset = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train_type1.csv', 
                      converters = {'Crime Code': crime_code,'Weapon Used Code':weapon})
dataset = dataset[np.isfinite(dataset['Victim Age'])]
dataset = dataset[np.isfinite(dataset['Weapon Used Code'])]

dataset['Victim Age'] = dataset['Victim Age'].replace([10,11,12,13,14,15,16,17,18,19],1)
dataset['Victim Age'] = dataset['Victim Age'].replace([20,21,22,23,24,25,26,27,28,29],2)
dataset['Victim Age'] = dataset['Victim Age'].replace([30,31,32,33,34,35,36,37,38,39],3)
dataset['Victim Age'] = dataset['Victim Age'].replace([40,41,42,43,44,45,46,47,48,49],4)
dataset['Victim Age'] = dataset['Victim Age'].replace([50,51,52,53,54,55,56,57,58,59],5)
dataset['Victim Age'] = dataset['Victim Age'].replace([60,61,62,63,64,65,66,67,68,69],6)
dataset['Victim Age'] = dataset['Victim Age'].replace([70,71,72,73,74,75,76,77,78,79],7)
dataset['Victim Age'] = dataset['Victim Age'].replace([80,81,82,83,84,85,86,87,88,89],8)
dataset['Victim Age'] = dataset['Victim Age'].replace([90,91,92,93,94,95,96,97,98,99],9)

dataset['Victim Sex'] = dataset["Victim Sex"].replace(['H','-','X'],'U')
dataset['Victim Sex'].value_counts()

dataset=dataset[dataset['Victim Sex'].notnull()]
dataset.shape
X_full = dataset.iloc[:,[11,12]].values
y_full = dataset.iloc[:, 8].values


#encoding values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_full[:, 1] = labelencoder_X.fit_transform(X_full[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_full = onehotencoder.fit_transform(X_full).toarray()
X_full = X_full[:,1:]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y_full = labelencoder_y_1.fit_transform(y_full)



# create model
model = Sequential()
model.add(Dense(150, input_dim=3, activation='relu'))
model.add(Dense(100, activation='tanh'))
#model.add(Flatten())
model.add(Dense(9, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#divide training data into 10 pairs of train and dev
#kfn = kf(n_splits=10, random_state=5, shuffle=False)
#kfn.get_n_splits(X_full)


#from sklearn.model_selection import StratifiedKFold
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
accuracies = []
#for train, test in kfold.split(X_full, y_full):
#    model.fit(X_full[train], y_full[train], epochs=150, batch_size=10, verbose=0)
#    # evaluate the model
#    scores = model.evaluate(X_full[test], y_full[test], verbose=0)
#    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#    accuracies.append(scores[1] * 100)


X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.25, random_state = 5)

model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
accuracies.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))

#i=0;
#for train_index, dev_index in kfn.spl+it(X_full):
#    #print("train:", train_index, "dev:", dev_index)
#    i +=1
#    print("In iteration: ")
#    print(i)
#    X_train, X_dev = X_full[train_index], X_full[dev_index]
#    y_train, y_dev = y_full[train_index], y_full[dev_index]
#
#    #train model    
#    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose = 1)
#    
#    scores = model.evaluate(X_dev, y_dev)
#    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#    accuracies.append(scores[1]*100)
