import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

csvData = pd.read_csv("Data.csv")
x = csvData.iloc[:, :-1].values
y = csvData.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
y_actual = y_test.reshape(len(y_test),1)


#Multivariate
linear = LinearRegression()
linear.fit(x_train, y_train)

y_multi_pred = linear.predict(x_test)

predicted = y_multi_pred.reshape(len(y_multi_pred),1)
print('Multivariate: ' + str(r2_score(y_actual, predicted)))


#Polynomial
poly = PolynomialFeatures(degree = 4)
x_poly = poly.fit_transform(x)
x_poly_train, x_poly_test = train_test_split(x_poly, test_size = 0.2, random_state = 0)

linear.fit(x_poly_train, y_train)
y_poly_pred = linear.predict(x_poly_test)
predicted = y_poly_pred.reshape(len(y_poly_pred),1)
print('Polynomial: ' + str(r2_score(y_actual, predicted)))


#Decision Tree
tree = DecisionTreeRegressor(random_state = 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
y_actual = y_test.reshape(len(y_test),1)

tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)
predicted = y_pred.reshape(len(y_pred),1)
print('Decision Tree: ' + str(r2_score(y_actual, predicted)))


#Random Forest
forest = RandomForestRegressor()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
y_actual = y_test.reshape(len(y_test),1)

forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
predicted = y_pred.reshape(len(y_pred),1)
print('Random Forest: ' + str(r2_score(y_actual, predicted)))


#Support Vector
sc_x = StandardScaler()
sc_y = StandardScaler()
x_vector = sc_x.fit_transform(x)
y_vector = sc_y.fit_transform(y.reshape(-1, 1)).ravel()
x_vector_train, x_vector_test, y_vector_train, y_vector_test = train_test_split(x_vector, y_vector, test_size = 0.2, random_state = 0)
y_vector_actual = sc_y.inverse_transform([y_vector_test]).flatten()

vector = SVR(kernel ='rbf')
vector.fit(x_vector_train, y_vector_train)

y_vector_pred = sc_y.inverse_transform([vector.predict(x_vector_test)]).flatten()
print('Support Vector: ' + str(r2_score(y_vector_actual, y_vector_pred)))