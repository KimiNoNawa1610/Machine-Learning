import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#Decision Tree
data1 = pd.read_csv("Position_Salaries.csv")

x = data1[["Level"]]
y = data1["Salary"]


print(x)
print(y)

regressor1 = DecisionTreeRegressor(random_state=0)

regressor1.fit(x,y)

y_pred1 = regressor1.predict([[6.5]])

print(y_pred1)

x_grid = np.arange(int(x.min()), int(x.max()), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plot.scatter(x,y, color="red")
plot.plot(x_grid,regressor1.predict(x_grid), color="blue")
plot.show()


#Random Tree
data1 = pd.read_csv("Position_Salaries.csv")

x = data1[["Level"]]
y = data1["Salary"]

regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor2.fit(x,y)

y_pred2 = regressor2.predict(np.array([6.5]).reshape(1,1))

x_grid = np.arange(int(x.min()), int(x.max()), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plot.scatter(x,y, color="orange")
plot.plot(x_grid,regressor1.predict(x_grid), color="green")
plot.show()