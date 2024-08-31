import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# Load the cleaned data
new_data = pd.read_csv('/Users/anithasmac/PycharmProjects/FinalProject/Model_Data.csv')

X = new_data.drop(columns=['Weekly_Sales'])
y = new_data['Weekly_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Setup the GridSearchCV
grid_search_random = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2)

# Fit the model
grid_search_random.fit(X_train, y_train)


# Print the best parameters and best score
print("Best parameters:", grid_search_random.best_params_)
print("Best score:", grid_search_random.best_score_)

