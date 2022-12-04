import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv("archive\Bank Customer Churn Prediction.csv")

lEncoder = LabelEncoder()

dataset.country = lEncoder.fit_transform(dataset.country)
dataset.gender = lEncoder.fit_transform(dataset.gender)

print(dataset.describe(include='all'))

print(dataset['churn'].value_counts())

x = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1:]
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("Random Forest accuracy score:",accuracy_score(y_test,y_pred_rf))

#SVC

svm = svm.SVC()

svm.fit(x_train,y_train)

y_pred_svm=svm.predict(x_test)

print("SVC accuracy score:",accuracy_score(y_test,y_pred_svm))

#KNeighbors Classifier

knn = KNeighborsClassifier()

knn.fit(x_train,y_train)

y_pred_knn=knn.predict(x_test)

print("Knn accuracy score:",accuracy_score(y_test,y_pred_knn))

#Decision Tree

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_pred_dt=dt.predict(x_test)

print("Decision Tree accuracy score:",accuracy_score(y_test,y_pred_dt))

#Logistic Regression

log = LogisticRegression()

log.fit(x_train, y_train)

y_pred_log = log.predict(x_test)

print("Logistical accuracy score:",accuracy_score(y_test,y_pred_log))

performance_summary = pd.DataFrame({
    'Model':['Random Forest','SVC','KNN','Decision Tree','Logistical'],
    'ACC':[accuracy_score(y_test,y_pred_rf),
           accuracy_score(y_test,y_pred_svm),
           accuracy_score(y_test,y_pred_knn),
           accuracy_score(y_test,y_pred_dt),
           accuracy_score(y_test,y_pred_log),
          ]
})

#Interpretation: We can see that random forest has the highest accuracy score of 5 models. This can be because Random forest
#improves on bagging because it decorrelates the trees with the introduction of splitting on a random subset of features fo the dataset