# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_csv('kidney_disease.csv')

#X denotes the dataset
X = data[['rbc','pc','pcc','ba','htn']]
# y denotes the classes or labels
y = data[['classification']]


print (X.isnull().sum())

#Check the unique values in the extracted columns:
print (pd.unique(X['rbc']))
print (pd.unique(X['pc']))
print (pd.unique(X['pcc']))
print (pd.unique(X['ba']))
print (pd.unique(X['htn']))

#Fill the missing values with generative string
X = X.fillna('other')

# Use the label encoder to transform the string labels to numeric labels
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X['rbc'] = le.fit_transform(X['rbc'])
X['pc'] = le.fit_transform(X['pc'])
X['pcc'] = le.fit_transform(X['pcc'])
X['ba'] = le.fit_transform(X['ba'])
X['htn'] = le.fit_transform(X['htn'])

#Divide the dataset in training and testing data
#step 1: call the train test split method
from sklearn.model_selection import train_test_split
#step 2: split the data using the train test split method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#Configure and train the KNN classifier
#step 1:call the knn model from the sklearn library
from sklearn.neighbors import KNeighborsClassifier
#step 2: configure the classifier with number of neighbors required 
#to take the classification decision
clf = KNeighborsClassifier(n_neighbors=3)
#step 3: Train the classifer
y_train = y_train.iloc[:,0]
clf.fit(X_train,y_train)
#step 4: test the classifier
y_preds = clf.predict(X_test)
#step 5: call the accuracy_score method
from sklearn.metrics import accuracy_score
#step 6: analyze the classification accuracy
print ('Accuracy of the propose model: ')
print (accuracy_score(y_preds,y_test))