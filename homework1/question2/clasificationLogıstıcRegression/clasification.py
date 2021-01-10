# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 03:01:56 2021

@author: Baykal
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("data.csv", sep=";")

print(data)

x = data.iloc[:,2:4].values

x = data.iloc[:,1:3].values
y = data.iloc[:,4:].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)







