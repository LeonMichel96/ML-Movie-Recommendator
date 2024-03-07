import os
import pandas as pd
import numpy as np
import streamlit as st

from utils import preprocessor, x_projected, merged_data, find_movies, nearest_n

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

# Page Header

st.markdown('''
        # Movie Recommendator App :tv:
    '''
)

st.subheader('Choose five of your favorite movies')
st.text('Note: some movies may not be included in the database due to prior cleaning steps')

# Movie selection and identification, movies variable just for test

with st.form('User input'):
    col1, col2, col3 = st.columns(3, gap='large')

    with col1:
        movie1 = st.selectbox("Choose first movie",
                    merged_df['Title'], index=135562)
    with col2:
        movie2 = st.selectbox("Choose second movie",
                    merged_df['Title'],index=295370)

    with col3:
        movie3 = st.selectbox("Choose third movie",
                    merged_df['Title'], index=65433)


    col4, col5 = st.columns(2, gap='large')
    with col4:
        movie4 = st.selectbox("Choose fourth movie",
                    merged_df['Title'], index=11335)
    with col5:
        movie5 = st.selectbox("Choose fifth movie",
                    merged_df['Title'], index=22440)

    submitted = st.form_submit_button("Submit")

if submitted:
    movies = [movie1, movie2, movie3, movie4, movie5]
    selected_df, movies_indexes = find_movies(merged_df, movies)

    print(selected_df)

# KNN steps

    recommendation_df = nearest_n(selected_df,X_projected,merged_df, movies_indexes)
    st.write(recommendation_df.head(11))
