import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
 
data = pd.read_csv("C:\\Users\\online\\Desktop\\codes\\iris.csv")
 
X =data.iloc[:,[1,2,3,4]].values
y =data.iloc[:,5].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
 
from sklearn.naive_bayes import GaussianNB
model_1 = GaussianNB()
from sklearn.linear_model import LogisticRegression
model_2 = LogisticRegression(max_iter = 1000)
from sklearn.ensemble import RandomForestClassifier
model_3 = RandomForestClassifier()
 
# training all the model on the training dataset
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)
 
# predicting the output on the validation dataset
pred_1 = model_1.predict(X_test)
pred_2 = model_2.predict(X_test)
pred_3 = model_3.predict(X_test)
 

from sklearn import metrics
FP1=metrics.accuracy_score(y_test, pred_1)*100
FP2=metrics.accuracy_score(y_test, pred_2)*100
FP3=metrics.accuracy_score(y_test, pred_3)*100

#        Averaging Method
print("Classification Accuracy of First Model:", FP1)
print("Classification Accuracy of Second Model:", FP2)
print("Classification Accuracy of Third Model:", FP3)
print("Averaging Method Classification Accuracy :",(FP1+FP2+FP3)/3) 

#       Maximum Voting
print("Maximum Voting Method Classification Accuracy :",max(FP1,FP2,FP3)) 
