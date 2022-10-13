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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


csvData = pd.read_csv("Data Midterm.csv")
x = csvData.iloc[:, :-1].values
y = csvData.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
y_actual = y_test.reshape(len(y_test),1)

#Feature scaling
sc=StandardScaler()

x_train = sc.fit_transform(x_train)
x_train

x_test = sc.transform(x_test)
x_test

# #Decision Tree
# tree = DecisionTreeRegressor(random_state = 0)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# y_actual = y_test.reshape(len(y_test),1)

# tree.fit(x_train, y_train)
# y_pred = tree.predict(x_test)
# predicted = y_pred.reshape(len(y_pred),1)
# print('Decision Tree: ' + str(r2_score(y_actual, predicted)))


# #Random Forest
# forest = RandomForestRegressor()
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# y_actual = y_test.reshape(len(y_test),1)

# forest.fit(x_train, y_train)
# y_pred = forest.predict(x_test)
# predicted = y_pred.reshape(len(y_pred),1)
# print('Random Forest: ' + str(r2_score(y_actual, predicted)))

#Logistic Class

#DT Class
#RF class

#Fitting Kernel SVM
k_SVM = SVC(kernel = 'rbf', random_state = 0)
k_SVM.fit(x_train, y_train)
k_SVM

y_pred = k_SVM.predict(x_test)
y_pred

#Confusion Matrix
cm = confusion_matrix(y_test, y_actual)
cm

accuracies = cross_val_score(estimator = k_SVM, X = x_train, y = y_train, cv = 10)
accuracies
accuracies.mean()
accuracies.std()

parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
parameters

grid_search = GridSearchCV(estimator = k_SVM,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
#grid_search

best_accuracy = grid_search.best_score_
best_accuracy

best_parameters = grid_search.best_params_
best_parameters


#KNN/SVM

