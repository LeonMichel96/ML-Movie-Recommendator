# Movie Recommendator System
This project consists in building a movie recommendator system based on the
IMDB public datasets.
- The user will provide as input his selection comming from IMDB datastes some of his most favorites movies.
- Using PCA techniques we will find new patterns in the features that will help the algorithm to group movies in a better way.
- Using K Nearest Neighbors the program will identify the 10 most similar titles for each movie selected.
- Finally K Means technique will allow us to group into three different clusters  the recommendations.


The final output will be a number of movie clusters
based on the user preferences.

Note: the raw datasets are too big to push to github, you can either download them manually and run the extraction steps using the extraction notebook, or just use the clean_df.csv that is already uploaded.

In case you want to follow all steps you will need to download locally the following files:

- title.ratings
- title.basics
- title.crews
- name.basics

Link: https://datasets.imdbws.com/
