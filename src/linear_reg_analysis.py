"""
Alec O'Connor
Linear Regression Analysis
Description: This program utilizes various linear regression models to
predict box office revenues for movies.
"""

import pandas as pd
import helper_functions as hf
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif


# Function call to ingest movie data from a .csv file into a dataframe
movie_df = hf.get_movie_df('tmdb_movie_data.csv')

# Splits the data in the movie dataframe into X and Y training and test data
X_train, X_test, Y_train, Y_test = hf.get_train_test_data(movie_df)

# Single feature linear regression analysis with movie popularity as the
# predictor variable, so creating X and Y series with only movie popularity data
X_train_single_feature = X_train['popularity']
X_test_single_feature = X_test['popularity']
# Instantiates the linear regression classifier as the model
lin_reg = LinearRegression()
# Fits the linear regression model to the movie popularity data
lin_reg.fit(X_train_single_feature.values.reshape(-1, 1), Y_train)
# Predicts the movie revenue based on only the movie popularity
Y_pred = lin_reg.predict(X_test_single_feature.values.reshape(-1, 1))
# Creates a plot of the movie popularity vs. movie revenue
hf.create_regression_plot(X_test_single_feature, Y_test, Y_pred,
                          'Predictor variable - Movie Popularity',
                          'Response Variable - Movie Revenue',
                          'Single Feature Linear Regression')
# Compute metrics from the single feature linear regression model
print('Single Feature Linear Regression Analysis with Movie Popularity Feature Results:')
hf.get_regression_metrics(Y_test, Y_pred, lin_reg)


# Multiple feature linear regression analysis with all features
# Drops all rows which contain NaN values
movie_df_new = movie_df.dropna()
# Gets new X and Y training data which does not include NaN values
X_train_multi, X_test_multi, Y_train_multi, Y_test_multi = hf.get_train_test_data(movie_df_new)
# Fits the linear regression model to all the features in the movie data
lin_reg.fit(X_train_multi, Y_train_multi)
# Predicts the movie revenue based on all the features in the movie dataframe
Y_pred_multi = lin_reg.predict(X_test_multi)
# Compute metrics from the multiple feature linear regression model
print('\nMultiple Feature Linear Regression Analysis with all Features Results:')
hf.get_regression_metrics(Y_test_multi, Y_pred_multi, lin_reg)


# K-best features linear regression analysis for k ranging from 1 to 5
print('\nK-best Features Linear Regression Analysis:')
# Iterates through k values of 1 to 5
for k in range(1, 6):
    # Instantiates the K-best feature selector
    selector = SelectKBest(f_classif, k=k)
    # Fits the feature selector to the data
    selector.fit(X_train_multi, Y_train_multi)
    # Gets the indexes of the best column features in the X and Y data
    best_columns = selector.get_support(indices=True)
    # Creates a dataframe containing only the K-best columns of movie data
    movie_df_best = movie_df_new.iloc[:, best_columns]
    # Adds the movie revenue data column to the dataframe
    revenue_column = movie_df_new['revenue']
    movie_df_best = pd.concat([movie_df_best, revenue_column], axis=1)
    # Function call to get the training data based on the created dataframe
    X_train_best, X_test_best, Y_train_best, Y_test_best = hf.get_train_test_data(movie_df_best)
    # Fits the k-best data with the linear regression model
    lin_reg.fit(X_train_best, Y_train_best)
    # Predicts the movie revenue based on the k-best data features
    Y_pred_best = lin_reg.predict(X_test_best)
    print('K-best (k=' + str(k) + ') Feature Linear Regression Results:')
    column_names = movie_df_new.columns.values.tolist()
    best_column_names = []
    for i in best_columns:
        best_column_names.append(column_names[i])
    print('K-best Feature(s):', best_column_names)
    # Computes metrics for the k-best features linear regression model
    hf.get_regression_metrics(Y_test_best, Y_pred_best, lin_reg)
    print()
