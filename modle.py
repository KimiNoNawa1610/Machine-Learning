import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

dataset = pd.read_csv("bank-direct-marketing-campaigns.csv")
jobs = dataset["job"].unique()
#print(dataset)

#print(len(x[0]))
lEncoder = LabelEncoder()
#convert string value into numeric format
dataset.job = lEncoder.fit_transform(dataset.job)
dataset.marital = lEncoder.fit_transform(dataset.marital)
dataset.education = lEncoder.fit_transform(dataset.education)
dataset.default = lEncoder.fit_transform(dataset.default)
dataset.housing = lEncoder.fit_transform(dataset.housing)
dataset.loan = lEncoder.fit_transform(dataset.loan)
dataset.contact = lEncoder.fit_transform(dataset.contact)
dataset.month = lEncoder.fit_transform(dataset.month)
dataset.day_of_week = lEncoder.fit_transform(dataset.day_of_week)
dataset.poutcome = lEncoder.fit_transform(dataset.poutcome)

x = dataset.iloc[:,:-1].values

#print(len(x[0]))

y = dataset.iloc[:,-1].values

print("Number of yes: ",list(y).count("yes"))

print("Number of no: ",list(y).count("no"))

#print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.25,random_state=0)

svc_model = SVC(kernel='rbf',random_state=0)#kernel Support Vector is very powerful

svc_model.fit(x_train,y_train)

y_pred = svc_model.predict(x_test)

#actual = lEncoder.fit_transform(y_test).reshape(len(y_pred),1)

#predictied = lEncoder.fit_transform(y_pred).reshape(len(y_pred),1)

#print(actual)

#print(predictied)

print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred))

print("Accuracy Score: ",svc_model.score(x_test, y_test))

plot_confusion_matrix(svc_model,x_test,y_test)

plt.show()


#Interpretation:
"""
The models has accuracy of 89%, which is relatively accurate. On the other hand, 
the confusion matrix showed that the true positive value is much more higher than false positive and false negative value, 
which implies the accuracy of the model. On more thing I noticed that the percentage of people say no is much higher than 
the percentage of people say yes. 
"""


