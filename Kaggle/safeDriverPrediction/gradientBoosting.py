#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 23:23:10 2017

@author: cedric
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import time

np.random.seed(0)
path_train = '/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/train.csv'
path_test = '/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/test.csv'

df_tot = pd.read_csv(path_train)
df_Data = pd.read_csv(path_test)
print df_tot[df_tot['id']==1063865]

def ModelGradientboosting(df = df_tot, df_Data = df_Data, sample = 550000, ID = 'id',target='target'):
    df_train=df.sample(n=sample,frac=None,random_state=0)
    df_test=df[~df[ID].isin(df_train[ID])]
    df_train.set_index(ID, inplace=True)
    df_test.set_index(ID, inplace=True)
   
    df_Ytrain = pd.DataFrame(df_train[target])
    df_Xtrain = df_train.drop(target, axis=1)

    df_Ytest = pd.DataFrame(df_test[target])
    df_Xtest = df_test.drop(target, axis=1)
    loss = ['exponential']
    n_estimators = [80]
    Learning_rate = [0.01]
    max_depth = [4]
    min_samples_split = [2]
    subsample = [1.0]
    max_features =['auto']
    L = []
    Dict = {}
    Start_Timer = time.clock()
    for ii in loss:
        for i in Learning_rate:
            for j in n_estimators: 
                for k in max_depth:
                    for m in min_samples_split:   
                        for n in subsample:
                            for o in max_features:
                                GBest = GradientBoostingClassifier(loss = ii,
                                                                    n_estimators=j,
                                                                   learning_rate=i,
                                                                   max_depth = k,
                                                                   min_samples_split = m,
                                                                   subsample = n,
                                                                   max_features = o,
                                                                   verbose=1).fit(df_Xtrain, df_Ytrain)
                                
                                nicolecaca=GBest.predict_proba(df_Xtest)
                                nicolegroscaca=pd.DataFrame(nicolecaca)
                                nicolegroscaca.rename(columns={0:'proba0', 1:'target'}, inplace= True)
                                df_Ytest['proba0Predicted'] = nicolegroscaca['proba0']
                                df_Ytest['Evaluation'] =(1-df_Ytest['target'])*df_Ytest['proba0Predicted'] - df_Ytest['target']*df_Ytest['proba0Predicted']
                                Score  = int(np.sum(df_Ytest['Evaluation']))*100.0/(len(df_Ytest)-int(np.sum(df_Ytest['target'])))
                                
                                print (Score,'loss=',ii,
                                             'learningRate=',i,
                                             'n_estimators=',j,
                                             'max_depth=',k,
                                             'min_samples_split=',m,
                                             'subsample=',n,
                                             'max_features=',o)                
                                Dict[(ii,i,j,k, m,n,o)] = Score
                                print "--- %s seconds --- " %(time.clock() - Start_Timer)
                    
    A = max(Dict, key=Dict.get)
    print Dict[A]
    GTrue = GBest  
    df_Data.set_index(ID, inplace=True)
    B = GTrue.predict_proba(df_Data)
    df_result = pd.DataFrame(B)
    df_result.rename(columns={0:'proba0', 1:'target'}, inplace= True)
    df_Final = pd.concat([df_result, pd.DataFrame(df_Data.index)], axis =1)
    del df_Final['proba0']
    df_Final.set_index(ID, inplace=True)
#    df_Final.to_csv('/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/result3Propre.csv', sep=',')       
    L.append(A)
    L.append(Dict)
    L.append(df_Final)
    print "--- %s seconds --- " %(time.clock() - Start_Timer)
    return L


Result = ModelGradientboosting()
# best one for now: 'exponential', 0.01, 80, 4, 2 1,0, auto