
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

variables = pd.read_csv("sales.csv", sep=";")

print(variables)

months = variables[['Months']]
print(months)

sales = variables[['Sales']]
print(sales)

sales2 = variables.iloc[:,:1].values
print(sales2)

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

prediction = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("sales by month")
plt.xlabel("Months")
plt.ylabel("Sales")














