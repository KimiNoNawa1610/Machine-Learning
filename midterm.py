import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


csvData = pd.read_csv("Data Midterm.csv")
x = csvData.iloc[:, :-1].values
y = csvData.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
y_actual = y_test.reshape(len(y_test),1)

#Feature scaling
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#Logistic Regression
Log_classifier = LogisticRegression(random_state=0)
Log_classifier.fit(x_train,y_train)
y_pred = Log_classifier.predict(x_test)


np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)
cm = confusion_matrix(y_test,y_pred)
print("Logistic Results:")
print(cm)
print(accuracy_score(y_test,y_pred))
print()


#KNN
k_neighbor = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
k_neighbor.fit(x_train, y_train)

y_pred = k_neighbor.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("K-NN Results:")
print(cm)
print(accuracy_score(y_test, y_pred))
print()


#SVM
svc = SVC(kernel = 'linear', random_state = 0)
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)
cm = confusion_matrix(y_test, y_pred)
print("SVM Results:")
print(cm)
print(accuracy_score(y_test, y_pred))
print()


#Kernel SVM
k_SVM = SVC(kernel = 'rbf', random_state = 0)
k_SVM.fit(x_train, y_train)
y_pred = k_SVM.predict(x_test)

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm

accuracies = cross_val_score(estimator = k_SVM, X = x_train, y = y_train, cv = 10)
print("Kernel SVM Results:\n")
print("cross validation:", accuracies)
accuracies.mean()
accuracies.std()

parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = k_SVM,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("Grid Search Results:", grid_search)
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
print()


#Decision Tree Classification
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT_classifier.fit(x_train, y_train)

y_pred = DT_classifier.predict(x_test)

print("DT Results:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print()


#Random Forest Classification
RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_classifier.fit(x_train, y_train)

y_pred = RF_classifier.predict(x_test)

print("RF Results:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print()


#XGBoost
import xgboost
from xgboost import XGBClassifier

y[y == 2] = 0
y[y == 4] = 1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

xgb_clf = XGBClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3)
xgb_clf.fit(x_train, y_train)

y_pred = xgb_clf.predict(x_test)

print("XGBoost Results:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print()




