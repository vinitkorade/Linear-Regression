# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:47:20 2019

@author: VINIT KORADE
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\LP3\\ML\\Exp1\\data.csv");
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size= 1/3, random_state= 1 )

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

print("\nCoefficients:",regressor.coef_)

from sklearn.metrics import r2_score
print("Accuracy Score: ", r2_score(Y_test, Y_pred))

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_test, Y_pred, color='blue')
plt.title("Training Dataset")
plt.xlabel("Number of Hours Spent Driving")
plt.ylabel("Risk Score")
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, Y_pred, color='blue')
plt.title("Test Dataset")
plt.xlabel("Number of Hours Spent Driving")
plt.ylabel("Risk Score")
plt.show()

plt.scatter(x, y, color='red')
plt.plot(X_test, Y_pred, color='blue')
plt.title("Dataset")
plt.xlabel("Number of Hours Spent Driving")
plt.ylabel("Risk Score")
plt.show()