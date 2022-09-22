# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.linear_model import LinearRegression

csvData = pd.read_csv("Position_Salaries.csv")

""" simplelinear """
x = csvData[["Level"]]
y = csvData["Salary"]

linear = LinearRegression()
linear.fit(x,y)
linearLine = plot.figure(1)


plot.scatter(x,y, color="red")
plot.plot(x,linear.predict(x),color ="green")
plot.title("Truth or Bluff (Simple Linear Regression)")
plot.xlabel("Position Level")
plot.ylabel("Salary")

y_pred_linear = linear.predict([[6.5]])

print(y_pred_linear)


""" polynomial """

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=5)
x_poly = poly.fit_transform(x)
linear.fit(x_poly,y)
polyCurve = plot.figure(2)
plot.scatter(x,y, color="red")
plot.plot(x,linear.predict(x_poly),color ="green")
plot.title("Truth or Bluff (Polynomial Regression)")
plot.xlabel("Position Level")
plot.ylabel("Salary")

y_pred_poly = linear.predict(poly.fit_transform([[6.5]]))

print(y_pred_poly)

plot.show()









# See PyCharm help at https://www.jetbrains.com/help/pycharm/
