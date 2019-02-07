#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:46:40 2019

@author: jarvis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:,3:13].values
Y = df.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_x_1 = LabelEncoder()
X[:,1] = label_x_1.fit_transform(X[:,1])
label_x_2 = LabelEncoder()
X[:,2] = label_x_2.fit_transform(X[:,2])
oneHotEnc = OneHotEncoder(categorical_features=[1])
X = oneHotEnc.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)   

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Make ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier = Sequential()
#Input layer and First hidden layer
classifier.add(Dense(6 , kernel_initializer='uniform',activation='relu',input_shape=(11,)))
#Implementing Dropout by disabling 10% of neurons Increasing rate when overfiting doesn't solve
classifier.add(Dropout(rate = 0.1))
#Hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
#Implementing Dropout by disabling 10% of neurons Increasing rate when overfiting doesn't solve
classifier.add(Dropout(rate = 0.1))
#Output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
#Compiling ANN
classifier.compile(optimizer=  'adam' , loss='binary_crossentropy', metrics=['accuracy'])
#Fitting Training Data into ANN
classifier.fit(X_train,Y_train,batch_size=10,epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Custom Prediction
new_pred = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred>0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , y_pred)

#Evaluating ANN
#Applying K-Fold Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6 , kernel_initializer='uniform',activation='relu',input_shape=(11,)))
    classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=  'adam' , loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

keras_classifier = KerasClassifier(build_fn=build_classifier , batch_size = 10 , epochs=100)
accuracies = cross_val_score(estimator= keras_classifier , X = X_train , y = Y_train , cv = 10 , n_jobs=-1)

mean = accuracies.mean()
varience = accuracies.std()

#Improving Accuracy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6 , kernel_initializer='uniform',activation='relu',input_shape=(11,)))
    classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer = optimizer , loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

keras_classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25 , 32],
              'epochs':[100 , 500],
              'optimizer':['adam' , 'rmsprop']}
grid_search = GridSearchCV(estimator = keras_classifier ,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train , Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_














