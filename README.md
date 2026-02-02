# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1.Start
Step-2.Import necessary libraries
Step-3.Load and preprocess the data (define features and target).
Step-4.Split the dataset into training and testing sets.
Step-5.Scale the features using StandardScaler.
Step-6.Train the SGDRegressor model on the training set.
Step-7.Evaluate the model on both training and testing sets using MSE or other metrics.
Step-8.End  

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: S.M.Kanishka
RegisterNumber:  212225220048

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X = data.data[:, :4]
Y = np.c_[data.target, data.data[:, 5]]

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)

sx, sy = StandardScaler(), StandardScaler()
Xtr, Xte = sx.fit_transform(Xtr), sx.transform(Xte)
Ytr, Yte = sy.fit_transform(Ytr), sy.transform(Yte)

model = MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3))
model.fit(Xtr, Ytr)

Ypred = sy.inverse_transform(model.predict(Xte))
Yte = sy.inverse_transform(Yte)

print("MSE:", mean_squared_error(Yte, Ypred))
print("Predictions:\n", Ypred[:5])
*/
```

## Output:
<img width="559" height="160" alt="image" src="https://github.com/user-attachments/assets/731384b5-6a2b-4afb-9e61-dd260c7df9ba" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
