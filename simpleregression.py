#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 23:47:49 2019

@author: enricotorriero
"""

https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import pandas as pd
#importing dataset
dataset = pd.read_excel(r'/Users/enricotorriero/Documents/Valuation/1st/regression/S&PxDELL/S&P-DELL_5Y.xlsx')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#splitting
x_train = x[:90]
x_test = x[90:]
y_train = y[:90]
y_test = y[90:]

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
#creat an object 
regression = linear_model.LinearRegression()

#training the model
regression.fit(x_train, y_train)

#making predictions 
y_pred = regression.predict(x_test)

#getting the coeficients, mean squared error and variance
print('Coefficients: \n', regression.coef_)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

#plotting the graph
import matplotlib.pyplot as plt
plt.scatter(x_test, y_test, color = 'black')
plt.plot(x_test, y_pred, color = 'blue')
plt.show()

