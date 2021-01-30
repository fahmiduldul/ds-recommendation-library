import numpy as np
import pandas as pd

def create_ranked_movies(movies_df: pd.DataFrame, reviews_df: pd.DataFrame):
    '''
    INPUT
    movies - the movies dataframe
    reviews - the reviews dataframe
    
    OUTPUT
    ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews, then time, and must have more than 4 ratings
    '''
    rating_mean = reviews_df.groupby('movie_id')['rating'].mean()
    rating_count = reviews_df.groupby('movie_id')['user_id'].count()
    rating_latest = reviews_df.groupby('movie_id')['timestamp'].max()
    rating_df = pd.concat([rating_mean, rating_count, rating_latest], axis=1)
    rating_df.columns = ['mean', 'count', 'latest_ts']

    ranked_movies = movies_df.merge(rating_df, left_on='movie_id', right_index=True)
    ranked_movies = ranked_movies.sort_values(["mean","count","latest_ts"], ascending=False)
    ranked_movies = ranked_movies[ranked_movies['count'] > 4][["movie", "mean","count","latest_ts"]]

    return ranked_movies

def popular_recommendation(n_top:int, ranked_movies:pd.DataFrame):
    '''
    INPUT:
    n_top - the number of recommendation returned
    ranked_movies - DataFrame of ranked movie

    OUTPUT:
    result - list of recommended movies name
    '''
    return list(ranked_movies['movie'].iloc[:n_top])

def find_similiar_movies(movie_id:int, movies_df:pd.DataFrame) -> str:
    '''
    INPUT:
    movie_id - movie id
    movies_df - movie DataFrame

    OUTPUT:
    result - name of the recommended movie
    '''
    #get row of given movie_id feature
    movie_mat = np.array(movies_df[movies_df['movie_id'] == movie_id].iloc[:,5:])[0]
    #get feature matrix of all movies
    movies_mat = np.array(movies_df.iloc[:,5:])

    #calculate similiarity between given movie and all movie
    dot_prod = movie_mat.dot(movies_mat.transpose())

    #get the most likely movie
    movie_rows = np.where(dot_prod == np.max(dot_prod))[0]
    movie_row: int = np.random.choice(movie_rows)
    movie = movies_df.iloc[movie_row]['movie']

    return movie