import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("50_startups.csv")# import the dataset
x = dataset.iloc[:,:-1].values# independent variables
y = dataset.iloc[:,-1].values# dependent variable
print(x)

#transform columns
ct = ColumnTransformer(transformers=[('ecoder',OneHotEncoder(),[3])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

#split the data into 80/20 train/test ratio
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#use linear regression
linearRegression = LinearRegression()
linearRegression.fit(x_train,y_train)

y_pred = linearRegression.predict(x_test)# perform prediction

#output the prediction with the test data
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
