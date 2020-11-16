
#importing libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score


#importing dataset
dataset= pd.read_csv('C:/Users/USER/Documents/ML_Course_9-11-20-main/Project/churn.csv')
x = dataset.iloc[:,1:]
y = dataset.iloc[:, 0]

#important analysis
dataset.info()
dataset.describe
dataset.isin(['?']).sum()
# Our label Distribution (countplot)
sns.countplot(dataset["Churn"])
print(dataset.groupby(by="Churn").count())
#feature scaling

#spliting into test and train set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30) 

#defining models: decision tree classification
from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=4)
#fitting Decision Tree to test and train set
classifier.fit(x_train, y_train)

#overfitting or underfitting
from sklearn.metrics import accuracy_score
print('train score:',classifier.score(x_train, y_train))
print('test score:',classifier.score(x_test, y_test))
# no bias for training set
trainbias =1-(classifier.score(x_train, y_train))
print('Train Bias',trainbias)
testbias= 1-(classifier.score(x_test, y_test))
print('Test Bias',testbias)

variance= testbias-trainbias
print("variance:",variance)


#predicting the test set result
y_pred =classifier.predict(x_test)

#confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
###86% accuracy
perc= (849+72)/1000
###################################################
print(classification_report(y_test,y_pred))
##############################################333333333
print("Precision = {}".format(precision_score(y_test, y_pred)))
print("Recall = {}".format(recall_score(y_test, y_pred)))
print("Accuracy = {}".format(accuracy_score(y_test, y_pred)))
print("F1 Score = {}".format(f1_score(y_test, y_pred)))

############################################3333333333333
#defining model: LogisticRegression
from sklearn.linear_model import LogisticRegression
logRegressor= LogisticRegression(solver='liblinear', C=0.5, random_state=0)

#fitting logistic Regressor in train set
logRegressor.fit(x_train, y_train)

#predicting the test set result
y_pred=logRegressor.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
###84% accuracy
logRegressor.score
#defining models: RandomForest
from sklearn.ensemble import RandomForestClassifier
RandClassifier= RandomForestClassifier(n_estimators=100, criterion= 'gini', random_state=0)

#fitting logistic Regressor in train set
RandClassifier.fit(x_train, y_train)

#predicting the test set result
y_pred= RandClassifier.predict(x_test)

#confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
##93%accuracy


