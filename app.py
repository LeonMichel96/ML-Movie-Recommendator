import os
import pandas as pd
import numpy as np

from utils import preprocessor, x_projected, merged_data, find_movies

# Get the original df
base_directory = os.getcwd()
movies_df = pd.read_csv(os.path.join('/',base_directory,'clean','clean_df.csv'))


# Preprocessing steps
movies_df_transformed = preprocessor.fit_transform(movies_df)
movies_df_transformed = pd.DataFrame(movies_df_transformed, columns=preprocessor.get_feature_names_out())
movies_df_transformed.rename(columns={
    'remainder__tconst': 'id',
    'remainder__titleType': 'Type',
    'remainder__primaryTitle':'Title',
    'remainder__Director':'Director'
    }, inplace=True)

X_projected = x_projected(movies_df_transformed)

merged_df = merged_data(movies_df, X_projected)

# Movie selection and identification, movies variable just for test
movies = ['1917','Inception','The Notebook','Rocky III','Star Wars: Episode III - Revenge of the Sith']
selected_df = find_movies(merged_df, movies)

print(selected_df.head())
