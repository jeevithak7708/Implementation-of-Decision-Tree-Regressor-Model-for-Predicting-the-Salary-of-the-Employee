# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

DEVELOPED BY : JEEVITHA K

REGISTER NO : 212225040149


## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries and load the salary dataset
2.Separate input features and target values, then convert categorical data into numerical form
3.Split the dataset into training and testing data, and train the Decision Tree Regressor model
4.Plot and display the Decision Tree Regressor using plot_tree()

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Sample dataset
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)

# Split features and target
X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)

# Create Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# Predict salary for the dataset or new levels
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

# Example: predict salary for a new employee at level 6.5
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

# Visualize the results (High-resolution curve)
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

```

## Output:

<img width="897" height="759" alt="Screenshot 2026-05-24 211146" src="https://github.com/user-attachments/assets/734ba093-6cd4-45fe-89ef-a1b138723fa5" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
