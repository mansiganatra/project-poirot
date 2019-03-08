import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import SGDClassifier

def crime_code(str1):
    return int(str1[0:1])
    
dataset = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv', converters = {'Crime Code': crime_code})
dataset = dataset[np.isfinite(dataset['Victim Age'])]

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
X = dataset.iloc[:,[11,12]].values
y = dataset.iloc[:, 8].values

#handling the categorical values in victim gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
    
#Train the model using the training sets
knn.fit(X_train, y_train)
    
#Predict the response for test dataset
y_pred = knn.predict(X_test)
    
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
#    print(cm)
print("Accuracy:")
print(ac)
print(classification_report(y_test, y_pred))
        

#cc = dataset[["Crime Code", "Victim Sex",'Victim Age']]
#top10crime = cc["Crime Code"].value_counts().head(10).index
## Choosing data that is included in the top 10 crimes (by selection)
#crimecc = cc.loc[cc_vg["Crime Code"].isin(top10crime)]
#
#cc_age_gender = crimecc.groupby(["Crime Code", "Victim Sex",'Victim Age']).size().reset_index(name="Count")
#cc_age_gender

#print(cm)


#handling null values in victim age
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 0:1])
#X[:, 0:1] = imputer.transform(X[:, 0:1])

#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X, y)
#classifier=SGDClassifier(loss="hinge", penalty="l2", max_iter=5)