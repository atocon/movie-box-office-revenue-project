"""
Alec O'Connor
Movie Correlation Matrix
Description: This program ingests movie data into a dataframe and creates
a correlation matrix of the movie data features to aid data analysis.
"""

import helper_functions as hf
import matplotlib.pyplot as plt
import seaborn as sns

# Function call to ingest movie data from a .csv file into a dataframe
movie_df = hf.get_movie_df('tmdb_movie_data.csv')
# Prints the movie dataframe for visualization purposes
print(movie_df)

# Creates and plots a correlation matrix for the movie dataframe features
movie_df_corr = movie_df.corr()
# Sets the color theme of the heatmap plot
colormap = sns.color_palette('magma', as_cmap=True)
plt.title(u'Correlation Matrix - Movie data', size=16)
# Plots movie data correlation matrix
sns.heatmap(movie_df_corr, annot=True, cmap=colormap)
plt.show()
