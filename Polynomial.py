# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.linear_model import LinearRegression

csvData = pd.read_csv("Position_Salaries.csv")

x = csvData[["Level"]]
y = csvData["Salary"]

linear = LinearRegression()
linear.fit(x,y)

y_pred_l = linear.predict(x)
LinearRegression(copy_X=True,fit_intercept=True,n_jobs=None,normalize=False)
""" simplelinear """

""" polynomial """

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=5)
x_poly = poly.fit_transform(x)
linear.fit(x_poly,y)

plot.scatter(x,y)
plot.plot(x,linear.predict(x_poly),color ="green")

plot.show()







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
