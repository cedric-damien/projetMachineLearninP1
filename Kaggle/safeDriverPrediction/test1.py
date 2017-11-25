# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

np.random.seed(0)
path = '/Users/cedric/Documents/Kaggle/safeDriverPrediction/train.csv'

df_tot = pd.read_csv(path)



def Model1(df = df_tot, sample = 1000, ID = 'id',target='target'):
    df_train=df.sample(n=sample,frac=None,random_state=0)
    df_test=df[~df[ID].isin(df_train[ID])]
    df_trainUnsupervised=df_train[~]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(df_train)
    distances, indices = nbrs.kneighbors(X)
    return df_test

def Model(dfnew,sample =)

    
    
        



