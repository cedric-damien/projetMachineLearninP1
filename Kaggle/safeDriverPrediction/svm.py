#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:23:53 2017

@author: cedric
"""
import pandas as pd
import numpy as np
#from sklearn.ensemble import GradientBoostingClassifier
import time
from sklearn import svm

np.random.seed(0)
path_train = '/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/train.csv'
path_test = '/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/test.csv'

df_tot = pd.read_csv(path_train)
df_Data = pd.read_csv(path_test)

def ModelSupportVectorMachine(df = df_tot, df_Data = df_Data, sample = 50000, ID = 'id',target='target'):
    df_train=df.sample(n=sample,frac=None,random_state=0)
    df_test=df[~df[ID].isin(df_train[ID])]
    df_train.set_index(ID, inplace=True)
    df_test.set_index(ID, inplace=True)
   
    df_Ytrain = pd.DataFrame(df_train[target])
    df_Xtrain = df_train.drop(target, axis=1)

    df_Ytest = pd.DataFrame(df_test[target])
    df_Xtest = df_test.drop(target, axis=1)

    Start_Timer = time.clock()
    GBest = svm.SVC(probability=True)
    
    GBest.fit(df_Xtrain, df_Ytrain)
                                
    nicolecaca=GBest.predict_proba(df_Xtest)
    nicolegroscaca=pd.DataFrame(nicolecaca)
    nicolegroscaca.rename(columns={0:'proba0', 1:'target'}, inplace= True)
    df_Ytest['proba0Predicted'] = nicolegroscaca['proba0']
    df_Ytest['Evaluation'] =(1-df_Ytest['target'])*df_Ytest['proba0Predicted'] - df_Ytest['target']*df_Ytest['proba0Predicted']
    Score  = int(np.sum(df_Ytest['Evaluation']))*100.0/(len(df_Ytest)-int(np.sum(df_Ytest['target'])))
    print Score
    
    GTrue = GBest  
    df_Data.set_index(ID, inplace=True)
    B = GTrue.predict_proba(df_Data)
    df_result = pd.DataFrame(B)
    df_result.rename(columns={0:'proba0', 1:'target'}, inplace= True)
    df_Final = pd.concat([df_result, pd.DataFrame(df_Data.index)], axis =1)
    del df_Final['proba0']
    df_Final.set_index(ID, inplace=True)
#    df_Final.to_csv('/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/result3Propre.csv', sep=',')       

    print "--- %s seconds --- " %(time.clock() - Start_Timer)
    return 1


#Result = ModelGradientboosting()
# (12.13989909998033, 'loss=', 'deviance', 'learningRate=', 0.01, 'n_estimators=', 500, 'max_depth=', 6, 'min_samples_split=', 2, 'subsample=', 1.0, 'max_features=', None)