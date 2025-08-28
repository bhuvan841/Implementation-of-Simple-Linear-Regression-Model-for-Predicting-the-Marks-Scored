# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: BHUVANESHWARAN M

RegisterNumber: 212223080008
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df= pd.read_csv('data.csv')

df.head()
df.tail()

X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
y_pred

y_test

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
### Head Values
<img width="167" height="241" alt="Screenshot 2024-08-16 154352-1" src="https://github.com/user-attachments/assets/ba57a6e9-aae3-4e08-a457-858978a89625" />

### Tail Values
<img width="178" height="235" alt="Screenshot 2024-08-16 154419-1" src="https://github.com/user-attachments/assets/14f39504-2127-4a7e-8af0-a5078e891e38" />

### X Values
<img width="160" height="557" alt="Screenshot 2024-08-16 152702" src="https://github.com/user-attachments/assets/42b9f63b-27db-45c3-9391-1a908c8af39d" />

### y Values
<img width="718" height="60" alt="Screenshot 2024-08-16 153116-1" src="https://github.com/user-attachments/assets/c3431775-ad6a-44c8-b48d-49e9e896310a" />

### Predicted Values
<img width="698" height="74" alt="Screenshot 2024-08-16 161908" src="https://github.com/user-attachments/assets/4bbb2dc1-66ec-4bc0-8a9c-2d8474d65612" />

### Actual Values
<img width="576" height="28" alt="Screenshot 2024-08-16 153301" src="https://github.com/user-attachments/assets/73f99fb1-5c70-4a68-9fa3-12ea38d048f7" />

### Training Set
<img width="562" height="455" alt="download (8)" src="https://github.com/user-attachments/assets/ce8fae81-7f0a-4110-950f-affd27f7b492" />

### Testing Set
<img width="562" height="455" alt="download (7)-1" src="https://github.com/user-attachments/assets/3eceb402-704b-4e78-aa03-2bc93ffbc7a6" />

### MSE, MAE and RMSE

<img width="258" height="66" alt="Screenshot 2024-08-16 153958-1" src="https://github.com/user-attachments/assets/08f5261e-3f8b-454f-a3dc-cd11b0127611" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
