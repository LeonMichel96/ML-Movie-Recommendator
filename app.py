import os
import pandas as pd
#import numpy as np
import streamlit as st

from utils import preprocessor, x_projected, merged_data, find_movies, nearest_n, clustered_movies, movie_by_cluster

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

# Movie selection and identification

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


# KNN steps

    recommendation_df = nearest_n(selected_df,X_projected,merged_df, movies_indexes)

# KMeans steps
    final_df = clustered_movies(recommendation_df)



# Group by cluster
    cluster_1 = movie_by_cluster(final_df, 0)
    cluster_2 = movie_by_cluster(final_df, 1)
    cluster_3 = movie_by_cluster(final_df, 2)

    mix1_col1, mix1_col2 = st.columns(2, gap='large')

    with mix1_col1:
        st.subheader('First Movie Mix :fire:')
        st.text('First Segment of movie recommendations:')
        st.write(cluster_1)

    with mix1_col2:
        st.subheader('Genres')
        st.write('')
        st.write('')
        genres1=pd.DataFrame(cluster_1['Genre'].unique(), columns=['Genre'])
        st.write(genres1)

    mix2_col1, mix2_col2 = st.columns(2, gap='large')

    with mix2_col1:
        st.subheader('Second Movie Mix :sparkler:')
        st.text('Second Segment of movie recommendations:')
        st.write(cluster_2)

    with mix2_col2:
        st.subheader('Genres')
        st.write('')
        st.write('')
        genres2=pd.DataFrame(cluster_2['Genre'].unique(), columns=['Genre'])
        st.write(genres2)

    mix3_col1, mix3_col2 = st.columns(2, gap='large')

    with mix3_col1:
        st.subheader('Third Movie Mix :collision:')
        st.text('Third Segment of movie recommendations:')
        st.write(cluster_3)

    with mix3_col2:
        st.subheader('Genres')
        st.write('')
        st.write('')
        genres3=pd.DataFrame(cluster_3['Genre'].unique(), columns=['Genre'])
        st.write(genres3)
