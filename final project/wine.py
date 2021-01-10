# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:27:50 2021

@author: Baykal
"""

#data

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("wine.csv", sep=";")

print(data)

#missingvalues

data.sample(10)
permits.sample(5)

missing_values_count = nlf_data.isnull().sum()

missing_values_count[0:10]

total_cells = np.product(nlf_data.shape)

total_missing = missing_values_count.sum()

(total_missing/total_cells) * 100

total_cells = np.product(sf_permits.shape)

total_missing = missing_values_count.sum()

(total_cells / total_missing)*100

missing_values_count[0:10]

nlf_data.dropna()

colomns_with_na_dropped = nlf_data.dropna(axis=1)
colomns_with_na_dropped.head()


print("Orijinal veri setinin sütunları : %d \n" % nlf_data.shape[1])
print("Eksik verileri çıkarıldıktan sonraki sütun sayısı : %d" % colomns_with_na_dropped.shape[1])

colomns_with_na_dropped = sf_permits.dropna(axis=1)
colomns_with_na_dropped.head()

print("Orginal veri setinin sutunları : %d \n" % sf_permits.shape[1])
print("Eksik değerler silindikten sonra sf_permits veri setindeki  sütün sayısı : %d" % colomns_with_na_dropped.shape[1])


subset_nlf_data = nlf_data.loc[:, 'EPA':'Season'].head()
subset_nlf_data

subset_nlf_data.fillna(0)

subset_nlf_data.fillna(method = 'bfill', axis = 0).fillna("0")


x = data.iloc[:,2:9].values
y = data.iloc[:,11:].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#Define appropriate evaluation metric for our case (classification).
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

#Train and evaluate Decision Trees and at least 2 different appropriate algorithm which you can choose from scikit-learn library.
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color='red')
plt.plot(x,r_dt.predict(X), color='blue')

plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))







