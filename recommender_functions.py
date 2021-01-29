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
    rating_count = reviews_df.groupby('movie_id')['rating'].count()
    rating_latest = reviews_df.groupby('movie_id')['timestamp'].max()
    rating_df = pd.DataFrame({"mean": rating_mean, "count": rating_count, "latest_ts": rating_latest})

    ranked_movies = movies_df.merge(rating_df, how='left', on='movie_id', left_index=True)
    ranked_movies.sort_values(["mean","count","latest_ts"], ascending=False, inplace=True)
    ranked_movies = ranked_movies[ranked_movies['count'] > 4][["movie", "mean","count","latest_ts"]]

    return ranked_movies

def popular_recommendation(n_top, ranked_movies):
    ranked_movies['movies'][:n_top]