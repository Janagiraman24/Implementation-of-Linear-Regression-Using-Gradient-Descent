# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Load necessary libraries for data handling, metrics, and visualization.

2.Load Data: Read the dataset using pd.read_csv() and display basic information.

3.Initialize Parameters: Set initial values for slope (m), intercept (c), learning rate, and epochs.

4.Gradient Descent: Perform iterations to update m and c using gradient descent.

5.Plot Error: Visualize the error over iterations to monitor convergence of the model.

## Program:
### Program to implement the linear regression using gradient descent.
### Developed by: Janagiraman.M
### RegisterNumber: 212224230101
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
        pass
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

X=(data.iloc[1:, :-2].values)
print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="901" height="162" alt="Screenshot 2025-08-28 141119" src="https://github.com/user-attachments/assets/d8a29a5e-6bbb-4393-b7dc-2f47758923cd" />
<img width="843" height="885" alt="Screenshot 2025-08-28 141132" src="https://github.com/user-attachments/assets/41bef114-7268-4d68-bbc0-7498ce472fc3" />
<img width="836" height="884" alt="Screenshot 2025-08-28 141148" src="https://github.com/user-attachments/assets/0ba44f86-f517-41de-a760-495fabaece79" />
<img width="840" height="878" alt="Screenshot 2025-08-28 141213" src="https://github.com/user-attachments/assets/8915daeb-a560-4f9b-86c5-12bc49f516e4" />
<img width="850" height="38" alt="Screenshot 2025-08-28 141224" src="https://github.com/user-attachments/assets/091a5b49-c027-43e6-a8b2-b90c5d57d298" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
