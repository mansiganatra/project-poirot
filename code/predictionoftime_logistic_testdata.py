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
def format_time(m):
   new_time = '0'
   time = m
   if(len(time)==1):
       new_time = "0" +time
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
   print(final_time)
def crime_code(str1):
    return int(str1[0:1])
def format_crime_code(str1):
    return int(str1[0:1])
def date_time_converter(value):
    if(isinstance(value, str)):
        date = pd.to_datetime(value, format='%m/%d/%Y')
        return date
df = pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train_type1.csv',converters={'Date Occurred': date_time_converter, 
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

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
b1 = regressor.coef_
b0 = regressor.intercept_

y_pred_train = regressor.predict(X_train)
y_pred_dev = regressor.predict(X_dev)
y_pred_test = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
mse_train = mean_squared_error(y_train,y_pred_train)
rmse_train = mse_train**0.5

mse_dev = mean_squared_error(y_dev,y_pred_dev)
rmse_dev = mse_dev**0.5

mse_test = mean_squared_error(y_test,y_pred_test)
rmse_test = mse_test**0.5

print("************************ RMSE: ********************************")
print("Train: " + str(rmse_train))
print("Validation: " + str(rmse_dev))
print("Test: " + str(rmse_test))    
print("***************************************************************")




