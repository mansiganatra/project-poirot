import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import datetime as dt
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import json

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

def date_time_converter(value):
    if(isinstance(value, str)):
        date = pd.to_datetime(value, format='%m/%d/%Y')
        return date

def format_crime_code(str1):
    return int(str1[0:1])


def apply_logistic_regression(X_train, y_train, X_dev, y_dev): 
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_dev = sc.fit_transform(X_dev)
    
    classifier = LogisticRegression(random_state = 5)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_dev)
    cm = confusion_matrix(y_dev, y_pred)
    ac = accuracy_score(y_dev, y_pred)
    cl = classification_report(y_dev, y_pred)
    
    print(ac)
   # print(cl)
    print(cm)

def apply_linear_regression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    b1 = regressor.coef_
    b0 = regressor.intercept_
    y_pred = regressor.predict(X_test)
    m1=y_pred.astype(int)
    print(X_test)
    ##c=int(y_pred)
    print(m1)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test,y_pred)
    rmse = mse**0.5
    print(rmse)
    
    
df=pd.read_csv('D:/Sem1/INF552/Project/Data/Final1/train.csv',converters={'Date Occurred': date_time_converter, 
                                            'Time Occurred': format_time,
                                            'Crime Code': format_crime_code})
columns = df.columns
b=df[['Date Occurred','Time Occurred','Crime Code', 'Area ID']]

##print(columns)
a=type(columns)
##print(a)


##print(b)
column_1 = df.ix[:,3]
f=type(column_1)
##print(f)
#o=pd.DataFrame({"dayofweek": column_1.dt.dayofweek})
Y=b['Time Occurred'].values
print(Y)



b['weekday'] = column_1.dt.dayofweek
print(b)
#X = b[['weekday','Crime Code']].values
X = b.iloc[:,[2,3, 4]].values   
print(X)
Y = b.iloc[:, 1].values
#Y=b[['Time Occurred']].values
print(Y)


from sklearn.model_selection import train_test_split
#X = X.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#apply_logistic_regression(X_train, y_train, X_test, y_test)
apply_linear_regression(X_train, y_train, X_test, y_test)

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#sc_X = StandardScaler()
#X_train = sc_X.transform(X_train)
#X_test = sc_X.transform(X_test)
#print(X_test)



