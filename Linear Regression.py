import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#Nhan Vo
#CECS 456 Sec 03 Fall 2022
#import the csv data
csvData = pd.read_csv("Salary_Data.csv")

#check the data is correctly reading
print(csvData)

#get the list of value from YearsExperience and Salary
x = csvData[["YearsExperience"]] #horizontal
y = csvData["Salary"] #vertical

#split the dataset to 80/20 ratio, 80 for train and 20 for test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#train the model with train data
linearRegression = LinearRegression()
linearRegression.fit(x_train,y_train)

#Predict the result
y_pred = linearRegression.predict(x_test)

#use matplotlib to visualize the prediction and the training data
"""
plot.scatter(x_train,y_train,color = "red")
plot.plot(x_train,linearRegression.predict(x_train),color = "blue")
plot.title("Salary vs Experience (Training set)")
plot.xlabel("Years of Experience")
plot.ylabel("Salary")
plot.show()
"""
#use matplotlib to visualize the prediction and the test data
plot.scatter(x_test,y_test,color = "red")
plot.plot(x_train,linearRegression.predict(x_train),color = "blue")
plot.title("Salary vs Experience (Training set)")
plot.xlabel("Years of Experience")
plot.ylabel("Salary")
plot.show()








