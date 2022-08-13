"""
Alec O'Connor
Polynomial Regression Analysis
Description: This program utilizes various polynomial regression models to
predict box office revenues for movies.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import helper_functions as hf

# Function call to ingest movie data from a .csv file into a dataframe
movie_df = hf.get_movie_df('tmdb_movie_data.csv')

# Splits the data in the movie dataframe into X and Y training and test data
X_train, X_test, Y_train, Y_test = hf.get_train_test_data(movie_df)

# Choose the budget as single feature since it was the best feature from k-best
# linear regression analysis
X_train_single_feature = X_train['budget']
X_test_single_feature = X_test['budget']

# Performs polynomial regression of varying degrees with the movie budget
# feature to predict the movie revenue
print('Single Feature Polynomial Regression Analysis with Movie Budget Feature:')
# Iterates from 2 to 6 to evaluate different polynomial degrees
for n in range(2, 7):
    # Polynomial fitting of training data to degree n to get weights
    weights = np.polyfit(X_train_single_feature, Y_train, n)
    # Creates a polynomial model with weights
    model = np.poly1d(weights)
    # Predicts movie revenue based on movie budget feature data
    Y_pred = model(X_test_single_feature)
    # Creates a scatter plot with original Y data with a line
    # of best fit representing the predicted Y values
    plt.title('Single Feature Polynomial Regression, n = ' + str(n))
    plt.ylabel('Response Variable - Movie Revenue')
    plt.xlabel('Predictor variable - Movie Budget')
    # A line of evenly spaced points
    polyline = np.linspace(0, 3.8e8, len(movie_df))
    # Plots actual data
    plt.plot(X_test_single_feature, Y_test, 'b.', label='True Movie Revenue')
    # Plots line of best fit based on weights
    plt.plot(polyline, model(polyline), 'r-', label='Predicted Movie Revenue')
    plt.legend()
    plt.show()
    # Calculates and displays various metrics for the model
    print('Single Feature Polynomial Regression n=' + str(n) + ' Results:')
    print('Weights:', weights)
    r2 = r2_score(Y_test, Y_pred)
    print('Coefficient of Determination (R2):', r2)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    print('Root Mean Square Error (RMSE):', rmse)
    mae = mean_absolute_error(Y_test, Y_pred)
    print('Mean Absolute Error (MAE):', mae)
    print()


# Multiple feature polynomial analysis with all movie features
print('Multiple Feature Polynomial Regression Analysis with all Features:')
# Drops all rows which contain NaN values
movie_df_new = movie_df.dropna()
# Gets new X and Y training data which does not include NaN values
X_train_multi, X_test_multi, Y_train_multi, Y_test_multi = hf.get_train_test_data(movie_df_new)
# Iterates from 2 to 6 to evaluate different polynomial degrees
for n in range(2, 7):
    # Instantiates the linear regression model
    model = linear_model.LinearRegression()
    # Instantiates the polynomial features object with degree n
    poly = PolynomialFeatures(n)
    # Transforms X data with n degree polynomial features object
    X_train_multi_new = poly.fit_transform(X_train_multi)
    X_test_multi_new = poly.fit_transform(X_test_multi)
    # Fits the model to the data
    model.fit(X_train_multi_new, Y_train_multi)
    # Uses the model to predict the box office revenue values of X test data
    Y_pred_multi = model.predict(X_test_multi_new)
    # Calculates and displays various metrics for the model
    r2 = r2_score(Y_test_multi, Y_pred_multi)
    print('Multiple Feature Polynomial Regression n=' + str(n) + ' Results:')
    print('Coefficient of Determination (R2):', r2)
    rmse = mean_squared_error(Y_test_multi, Y_pred_multi, squared=False)
    print('Root Mean Square Error (RMSE):', rmse)
    mae = mean_absolute_error(Y_test_multi, Y_pred_multi)
    print('Mean Absolute Error (MAE):', mae)
    print()
