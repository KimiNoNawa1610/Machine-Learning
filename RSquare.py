import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Data.csv")


x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,4:5].values

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#VectorRegression
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

Vregressor = SVR(kernel ='rbf')
Vregressor.fit(x_train,y_train)

y_pred_V = sc_y.inverse_transform([Vregressor.predict(sc_x.transform(x_test))])

v_r2 = r2_score(sc_y.inverse_transform(y_test).flatten().tolist(),y_pred_V.tolist()[0])
print("VectoreRegression R2 score: ",v_r2)

#Decision Tree
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,4:5].values

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

DTregressor= DecisionTreeRegressor(random_state=0)

DTregressor.fit(x_train,y_train)

y_pred_DT = DTregressor.predict(x_test)

dt_r2 = r2_score(y_test.flatten().tolist(),y_pred_DT.tolist())
print("DecisionTree R2 score: ",dt_r2)

#Random Tree
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,4:5].values

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

RTregressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

RTregressor.fit(x_train,y_train)

y_pred_RT = RTregressor.predict(x_test)

rt_r2 = r2_score(y_test.flatten().tolist(),y_pred_RT.tolist())
print("DecisionTree R2 score: ",rt_r2)

#Multivariable Linear





















