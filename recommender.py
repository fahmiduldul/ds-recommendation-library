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
    learning_rate: float = 0.01, iters: int = 50) -> None:
        '''
        Create recommendation model

        INPUT:
        reviews_path - path of reviews data file
        movies_path - path of movies data file
        latent_features - number of latent features used
        learning_rate - gradeint boost learning rate
        iters - number of iteration
        '''
        
        #save parameter data
        self.reviews_df = pd.read_csv(reviews_path)
        self.movies_df = pd.read_csv(movies_path)
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        #wrangle reviews dataframe to dummy matrix
        user_rating_df = self.reviews_df.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        user_rating_mat = np.array(user_rating_df)
        n_users = user_rating_df.shape[0]
        n_movies = user_rating_df.shape[1]

        #save data for indexing
        self.users_arr = user_rating_df.index
        self.movies_arr = user_rating_df.columns

        # initialiaze U and V matrix
        u_mat = np.random.rand(n_users, self.latent_features)
        v_mat = np.random.rand(self.latent_features, n_movies)

        print("Interations \t | MSE")

        for iteration in range(iters):
            sse_cum = 0
            #iterate through all user_rating_df value
            for i in range(n_users):
                for j in range(n_movies):
                    #if the user give review, then do calculation
                    if user_rating_mat[i,j] > 0:
                        diff = user_rating_mat[i,j] - u_mat[i,:].dot(v_mat[:,j])
                        sse_cum += diff**2

                        #update u and v based with learning rate
                        for k in range(self.latent_features):
                            u_mat[i,k] += self.learning_rate * 2 * diff * v_mat[k,j]
                            v_mat[k,j] += self.learning_rate * 2 * diff * u_mat[i,k]
            print("{} \t\t | {}".format(iteration + 1, sse_cum))
        
        self.u_mat, self.v_mat = u_mat, v_mat


    def make_recs(self, id: str):
        return ['asd']

    def predict_rating(user_id, movie_id):
        pass