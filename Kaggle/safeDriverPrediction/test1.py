# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

np.random.seed(0)
path = '/Users/cedric/Documents/Kaggle/safeDriverPrediction/train.csv'

df_tot = pd.read_csv(path)



def Model1(df = df_tot, sample = 1000):
    df_train=df.sample(n=sample,frac=None,random_state=0)
    df_test=df[~df['id'].isin(df_train['id'])]
    return df_test


    
    
        



