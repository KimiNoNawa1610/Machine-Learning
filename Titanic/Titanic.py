#The following solution was used as a reference to build this model.
#https://www.kaggle.com/code/corazon17/titanic-data-analysis-and-classification
#https://www.kaggle.com/code/konohayui/titanic-data-visualization-and-models 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


import warnings
warnings.filterwarnings("ignore")

trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')
print(trainData.info())

print(f"\nTrain data has {trainData.duplicated().sum()} duplicated data")
print(f"Test data has {testData.duplicated().sum()} duplicated data")

#Data Cleaning and Feature Selection
#PassengerId
trainData.drop("PassengerId", axis=1, inplace=True)
testData.drop("PassengerId", axis=1, inplace=True)


#Modify Name to only contain Titles
trainData["Title"] = trainData["Name"].str.extract('([A-Za-z]+)\.')
testData["Title"] = testData["Name"].str.extract('([A-Za-z]+)\.')

trainData["Title"].value_counts()

def convert_title(title):
    if title in ["Ms", "Mile", "Miss"]:
        return "Miss"
    elif title in ["Mme", "Mrs"]:
        return "Mrs"
    elif title == "Mr":
        return "Mr"
    elif title == "Master":
        return "Master"
    else:
        return "Other"
        
trainData["Title"] = trainData["Title"].map(convert_title)
testData["Title"] = testData["Title"].map(convert_title)
trainData.drop("Name", axis=1, inplace=True)
testData.drop("Name", axis=1, inplace=True)


#Ticket
trainData.drop("Ticket", axis=1, inplace=True)
testData.drop("Ticket", axis=1, inplace=True)

#Cabin
trainData.drop("Cabin", axis=1, inplace=True)
testData.drop("Cabin", axis=1, inplace=True)

#Age
trainData.groupby('Title')['Age'].mean()
trainData.groupby('Title')['Age'].mean()
data = [trainData, testData]
for df in data:
    df.loc[(df["Age"].isnull()) & (df["Title"]=='Master'), 'Age'] = 5
    df.loc[(df["Age"].isnull()) & (df["Title"]=='Miss'), 'Age'] = 22
    df.loc[(df["Age"].isnull()) & (df["Title"]=='Mr'), 'Age'] = 32
    df.loc[(df["Age"].isnull()) & (df["Title"]=='Mrs'), 'Age'] = 36
    df.loc[(df["Age"].isnull()) & (df["Title"]=='Other'), 'Age'] = 44

#Fare
testData.Fare.fillna(trainData.groupby("Pclass").mean()["Fare"][3], inplace=True)

#SibSp and Parch
data = [trainData, testData]
for df in data:
    df['Relatives'] = df['SibSp'] + df['Parch']
    df.loc[df['Relatives'] > 0, 'Alone'] = 1
    df.loc[df['Relatives'] == 0, 'Alone'] = 0
trainData.drop(["SibSp", "Parch"], axis=1, inplace=True)
testData.drop(["SibSp", "Parch"], axis=1, inplace=True)

#Encode Categorical Features
trainData = pd.get_dummies(trainData, prefix=["Sex", "Embarked", "Title"])
testData = pd.get_dummies(testData, prefix=["Sex", "Embarked", "Title"])


#Model Building and Evaluation
x_train = trainData.drop("Survived", axis = 1)
y_train = trainData.Survived
x_test = testData.copy()

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifiers = {
    "KNN": KNeighborsClassifier(), 
    "LR": LogisticRegression(max_iter=1000), 
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "SVM": SVC(),
    "NB": GaussianNB()
}

results = pd.DataFrame(columns=["Classifier", "Avg_Accuracy"])
for name, clf in classifiers.items():
    model = clf
    cv_results = cross_validate(
        model, x_train, y_train, cv=10,
        scoring=(['accuracy'])
    )

    results = results.append({
        "Classifier": name,
        "Avg_Accuracy": cv_results['test_accuracy'].mean()
    }, ignore_index=True)
results = results.sort_values("Avg_Accuracy", ascending=False)
print(results)


#Using model with best accuracy score
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

results = testData.copy()
results["Survived"] = y_pred

print(testData.info())

#Pclass visualization
fig, ax = plt.subplots(1, 2, figsize = (9, 4))
results["Pclass"].value_counts().plot.bar(color = "green", ax = ax[0])
ax[0].set_title("Number of Passengers by Pclass")
ax[0].set_ylabel("Count")
sns.countplot(x = "Pclass", hue = "Survived", data = results, ax = ax[1])
ax[1].set_title("Survived vs Dead (Pclass)")
plt.show()

#Age visualization
fig, ax = plt.subplots(1, 2, figsize = (9, 5))
results[results["Survived"] == 0]["Age"].plot.hist(ax = ax[0], bins = 20, edgecolor = "black", color = "blue")
ax[0].set_title("Died (Age)")
domain_1 = list(range(0, 85, 5))
ax[0].set_xticks(domain_1)
results[results["Survived"] == 1]["Age"].plot.hist(ax = ax[1], bins = 20, edgecolor = "black", color = "green")
ax[1].set_title("Survived (Age)")
domain_2 = list(range(0, 85, 5))
ax[1].set_xticks(domain_2)
plt.show()

#Sex visualization
fig, ax = plt.subplots(1, 2, figsize = (9, 4))
results["Sex_female"].value_counts().plot.bar(color = "skyblue", ax = ax[0])
ax[0].set_title("Number Of Passengers By Sex")
ax[0].set_ylabel("Population")
sns.countplot(x="Sex_female", hue = "Survived", data = results, ax = ax[1])
ax[1].set_title("Sex: Survived vs Dead")
ax[1].set_xticklabels(['Male', 'Female'])
plt.show()