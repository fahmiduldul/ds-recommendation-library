import numpy as np
import pandas as pd
import sys

class Recommender():
    '''
    Class to give recommendation based on given matrix
    '''
    def __init__(self):
        pass

    def fit(self, reviews_path: str, movies_path: str, latent_features: int = 12,
    learning_rate: float = 0.005, iters: int = 20):
        #save parameter data
        self.reviews_df = pd.read_csv(reviews_path)
        self.movies_df = pd.read_csv(movies_path)
        self.n_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        user_rating_df = self.reviews_df.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        n_users = user_rating_df.shape[0]
        n_movies = user_rating_df.shape[1]

        # initialiaze U and V matrix
        u = np.random.random.rand(n_users, self.n_features)
        v = np.random.random.rand(self.n_features, n_movies)

        sse_cum = 0

        print("Interations | MSE")

        #iterate through all user_rating_df value
        for i in range(n_users):
            for j in range(n_movies):
                #if the user give review, then do calculation
                if user_rating_df[i,j] > 0:
                    diff = user_rating_df[i,j] - np.dot(u[i:,:], v[:,j])
                    sse_cum += diff**2

                    #update u and v based with learning rate
                    for k in range(self.n_features):
                        u[i,k] += self.learning_rate * 2 * diff * v[k,j]
                        v[k,j] += self.learning_rate * 2 * diff * u[i,k]


    def make_recs(self, id: str):
        return ['asd']

    def predict_rating(user_id, movie_id):
        pass