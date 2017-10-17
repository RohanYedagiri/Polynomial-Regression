# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

# dataset
dataset = pd.read_csv('Position_Salaries.csv')

# assign x & y variables
'''
Level column is like the encoded version of Position column, so we don't need to consider
Position
'''
X = dataset.drop(['Position','Salary'],axis=1)
y = dataset.Salary

# In order to compare Linear, Polynomial Regression we'll fit data to both models

# 1. Linear Regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)

# 2. Poly Regression
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree=4)
X_poly = polyreg.fit_transform(X)
X_poly = pd.DataFrame(X_poly)

# fit 2nd lin reg to x_poly
linreg2 = LinearRegression()
linreg2.fit(X_poly,y)

# visualize lin reg results
plt.scatter(X,y,color='red')
plt.plot(X, linreg.predict(X),color='blue')
plt.title('linear regression 1 predictions')
plt.xlabel('Position level')
plt.ylabel('salaries')
plt.show()


# visualize poly reg results
plt.scatter(X,y,color='red')
plt.plot(X, linreg2.predict(X_poly),color='blue')
plt.title('linear regression 1 predictions')
plt.xlabel('Position level')
plt.ylabel('salaries')
plt.show()

# predict lin reg
linreg.predict(6.5) # 330378.78 not the best

# predict poly reg
linreg2.predict(polyreg.fit_transform(6.5)) # 158862.45





