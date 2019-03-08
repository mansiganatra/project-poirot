import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from keras.models import Sequential
from keras.layers import Dense
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
    
dataset = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train_type1.csv', converters = {'Crime Code': crime_code,'Victim Age':age})
dataset = dataset[np.isfinite(dataset['Victim Age'])]
dataset = dataset[np.isfinite(dataset['Weapon Used Code'])]

dataset['Victim Sex'] = dataset["Victim Sex"].replace(['H','-','X'],'U')
dataset['Victim Sex'].value_counts()
dataset = dataset[dataset['Victim Descent'].notnull()]
dataset=dataset[dataset['Victim Sex'].notnull()]
dataset.shape

train = pd.concat([dataset.get(['Crime Code']),
                           pd.get_dummies(dataset['Victim Sex'], prefix='Gender'),
                           pd.get_dummies(dataset['Victim Age'], prefix='Age'),
                           pd.get_dummies(dataset['Type 1'],prefix='Type')],axis=1)

X = train.iloc[:,1:22].values
y = train.iloc[:, [22,23,24]].values

#handling the categorical values in victim gender
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:,1:]


model = Sequential()
model.add(Dense(150, input_dim=21, activation='relu'))
model.add(Dense(100, activation='tanh'))
#model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)

model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)
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
