#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 18:28:19 2018

@author: lovishdogra
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Salary_Data.csv')
# Independent variable(matrix of features)
X = dataset.iloc[:,:-1].values
# Dependent variable
y = dataset.iloc[:,1].values

# Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3,
                                                    random_state = 0)

# Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression as lp
regressor = lp()
regressor.fit(X_train, y_train)

# Predicting the training set results
y_pred = regressor.predict(X_test)

# Visualizing the training set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set results
y_pred = regressor.predict(X_test)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


























































































