# Imports
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing Pipelines
cat_transformer = Pipeline([
    ('Encoder',OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ('Scaler', StandardScaler())
    ])
num_transformer = Pipeline([('Scaler',StandardScaler())])

num_columns = ['isAdult', 'startYear', 'runtimeMinutes', 'averageRating', 'numVotes']
cat_columns = ['genres']

preprocessor = ColumnTransformer([
    ('cat_transformer', cat_transformer, cat_columns),
    ('num_transformer',num_transformer, num_columns)],
    remainder='passthrough'
)

#PCA functions
def x_projected(prepro_df):
    '''Fits a PCA object into the input dataframe and returns a projected dataframe in the new PC space.'''
    pca = PCA(n_components=20)
    X = prepro_df.drop(columns=['id', 'Type', 'Title', 'Director'])
    pca.fit(X)

    X_projected = pca.transform(X)
    X_projected = pd.DataFrame(X_projected, columns=[f'PC{i}' for i in range(1,21)])

    return X_projected

def merged_data(original_df, x_projected):
    '''
    Returns a merged dataframe combining the original columns of "Director", "Title", "Type" and "Genre"
    with the new projected dataframe in the PC space.

    '''
    identificator_df = original_df[['primaryTitle', 'genres', 'Director']]
    merged = identificator_df.merge(x_projected, how='left',left_index=True, right_index=True)
    merged.rename(columns={
        'primaryTitle':'Title',
        'genres':'Genre'
        }, inplace = True)

    return merged

# K Nearest Neighbors steps

def find_movies(merged_df, movies):
    '''
    Searches the indexes of the selected movies in the merged dataframe and returns a new dataframe of the
    projection in PC space of just the selected movies.

    '''
    movies_indexes = []
    for movie in movies:
        index = merged_df[merged_df['Title']==movie].index[0]
        movies_indexes.append(index)

    selected_movies = merged_df.iloc[movies_indexes]
    selected_movies.drop(columns=['Title', 'Genre', 'Director'], inplace =True)
    return selected_movies, movies_indexes

def nearest_n(selected_df, x_projected_df, merged_df, indexes):
    '''
    Fits a NearestNeighbors model and returns a dataframe with the original selected movies plus
    the 10 nearest neighbors of each one of them.

    '''
    nearest = NearestNeighbors(n_neighbors=11)
    nearest.fit(x_projected_df)

# Get 10 nearest neighbors of each movie
    k = nearest.kneighbors(selected_df, return_distance=False)

# Get all 55 indexes in k
    expanded_indexes = []
    for neighbor in k:
        for index_num in neighbor:
                expanded_indexes.append(index_num)

# Create final df with original selected movies and recommendations

    recomendation_df = merged_df.iloc[expanded_indexes]
    recomendation_df.drop(index=indexes, axis=0,inplace=True)
    recomendation_df.reset_index(inplace=True)
    recomendation_df.drop(columns=['index'], inplace=True)

    return recomendation_df
