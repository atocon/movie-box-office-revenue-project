"""
Alec O'Connor
Helper Functions
Description: Helper functions which are used throughout the project to
help predict movie box office revenue using different regression models.
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def get_movie_df(filename):
    """
    Ingests movie data from a .csv file and loads the data into a dataframe
    object.
    str:param filename: The name of the .csv file which contains movie data.
    df:return: A dataframe containing the movie data.
    """
    # Reads the movie data from the .csv file into a dataframe
    movie_df = pd.read_csv(filename)
    # Function call to add a column of labels indicating if the movie is in a
    # collection or not
    add_collection_labels(movie_df)
    # Function call to add a column of labels indicating which language the movie
    # is recorded in
    add_language_label(movie_df)
    # Function call to add a column of labels indicating which month the movie
    # was released in
    add_release_month_label(movie_df)
    # Instantiates a dataframe containing only specific columns of data
    movie_df = movie_df[['collection label', 'budget', 'language label',
                         'popularity', 'release month label', 'runtime', 'revenue']]
    return movie_df


def add_collection_labels(movie_df):
    """
    Adds a column of labels indicating whether or not a particular movie is
    in a collection or not to a dataframe containing movie data.
    df:param movie_df: A dataframe containing movie data.
    """
    # An empty list is instantiated to hold collection labels
    collection_labels = []
    # Gets a series of the belongs to collection data from the movie dataframe
    collection_df = movie_df['belongs_to_collection']
    # Creates a list of boolean values where null values are
    collection_null_values = collection_df.isnull()
    # Iterates over the rows of the collection data series
    for i in collection_null_values:
        if i is True:
            # The collection label is 0 if the movie is not part of a collection
            collection_labels.append(0)
        else:
            # The collection label is 1 if the movie is part of a collection
            collection_labels.append(1)
    # Adds the collection labels as a column to the dataframe
    movie_df['collection label'] = collection_labels


def add_language_label(movie_df):
    """
    Adds a column of labels indicating which language a movie was originally
    filmed in.
    df:param movie_df: A dataframe containing movie data.
    """
    # An empty list is instantiated to hold language labels
    language_labels = []
    # Gets a series of the original language data from the movie dataframe
    language_df = movie_df['original_language']
    # Iterates over the series object
    for i in language_df:
        if i == 'en':
            # The language label is 1 if the film is in English
            language_label = 1
        elif i == 'fr':
            # The language label is 2 if the film is in French
            language_label = 2
        elif i == 'ko':
            # The language label is 3 if the film is in Korean
            language_label = 3
        elif i == 'ru':
            # The language label is 4 if the film is in Russian
            language_label = 4
        elif i == 'ja':
            # The language label is 5 if the film is in Japanese
            language_label = 5
        else:
            # The language label is 6 in all other cases
            language_label = 6
        # Appends the language label to the empty list
        language_labels.append(language_label)
    # Adds the language label column to the movie dataframe
    movie_df['language label'] = language_labels


def add_release_month_label(movie_df):
    """
    Adds a column of labels to a movie dataframe indicating which month the
    movie was released in.
    df:param movie_df: A dataframe containing movie data.
    """
    # An empty list is instantiated to hold release month labels
    release_month_labels = []
    # Gets a series of the release date data from the movie dataframe
    release_date_df = movie_df['release_date']
    # Iterates over the series object
    for i in release_date_df:
        # Splits each release data into a list
        date_list = i.split('/')
        # Uses the month number as the release month label
        release_month_label = int(date_list[0])
        # Adds the release month label to the empty list
        release_month_labels.append(release_month_label)
    # Adds the release month column to the movie dataframe
    movie_df['release month label'] = release_month_labels


def get_train_test_data(movie_df):
    """
    50/50 Splits data in a movie dataframe into X and Y data to be used
    by regression models for predicting box office revenue of various movies.
    df:param movie_df: A dataframe containing movie data.
    df:return: Series objects representing X and Y training and testing data.
    """
    # Creates a dataframe of features which will be used to predict movie revenue
    X = movie_df.drop(columns=['revenue'])
    # Creates a series using the revenue column from the movie dataframe
    Y = movie_df['revenue']
    # Splits the data 50/50 into training and testing data sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=3)
    # Returns the data series objects
    return X_train, X_test, Y_train, Y_test


def get_regression_metrics(Y_test, Y_pred, model):
    """
    Calculates various metrics related to regression models and displays
    the information on the console.
    df:param Y_test: True value movie revenue data.
    df:param Y_pred: Predicted movie revenue data.
    model:param model: A regression model.
    """
    # Displays the model coefficients and y-intercept
    print('Weight Coefficient(s):', model.coef_)
    print('Y-Intercept:', model.intercept_)
    # Calculates and displays the coefficient of determination
    r2 = r2_score(Y_test, Y_pred)
    print('Coefficient of Determination (R2):', r2)
    # Calculates and displays the Root Mean Sqaure Error (RMSE)
    rmse = mean_squared_error(Y_test, Y_pred, squared=False)
    print('Root Mean Square Error (RMSE):', rmse)
    # Calculates and displays the Mean Absolute Error (MAE)
    mae = mean_absolute_error(Y_test, Y_pred)
    print('Mean Absolute Error (MAE):', mae)


def create_regression_plot(X_test, Y_test, Y_pred, x_label, y_label, title):
    """
    Creates a simple linear regression plot for single feature regression only.
    df:param X_test: Single feature data for testing.
    df:param Y_test: True value movie revenue data.
    df:param Y_pred: Predicted movie revenue data.
    str:param x_label: X-axis label for the plot.
    str:param y_label: Y-axis label for the plot.
    str:param title: Title for the plot.
    """
    # Plots the original data points
    plt.plot(X_test, Y_test, 'b.', label='True Movie Revenue')
    # Plots the predicted movie revenue values
    plt.plot(X_test, Y_pred, 'r-', label='Predicted Movie Revenue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
