# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import time


np.random.seed(0)
path_train = '/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/train.csv'
path_test = '/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/test.csv'

df_tot = pd.read_csv(path_train)
df_Data = pd.read_csv(path_test)
print df_tot[df_tot['id']==1063865]

def ModelGradientboosting(df = df_tot, df_Data = df_Data, sample = 300000, ID = 'id',target='target'):
    df_train=df.sample(n=sample,frac=None,random_state=0)
    df_test=df[~df[ID].isin(df_train[ID])]
    df_train.set_index(ID, inplace=True)
    df_test.set_index(ID, inplace=True)
   
    df_Ytrain = pd.DataFrame(df_train[target])
    df_Xtrain = df_train.drop(target, axis=1)

    df_Ytest = pd.DataFrame(df_test[target])
    df_Xtest = df_test.drop(target, axis=1)
    n_estimators = [80]
    Learning_rate = [ 0.01]
    max_depth = [4]
    min_samples_split = [2]
    L = []
    Dict = {}
    Start_Timer = time.clock()
    for i in Learning_rate:
        for j in n_estimators: 
            for k in max_depth:
                for m in min_samples_split:            
                    GBest = GradientBoostingClassifier(n_estimators=j,
                                                       learning_rate=i,
                                                       max_depth = k,
                                                       min_samples_split = m,
                                                       verbose=1).fit(df_Xtrain, df_Ytrain)
                    
                    df_Ytest['targetPredicted'] = GBest.predict(df_Xtest)
                    df_Ytest['Evaluation'] = np.square(df_Ytest['targetPredicted'] - df_Ytest['target'])
                    Score  = int(np.sum(df_Ytest['Evaluation']))*100.0/len(df_Ytest['Evaluation'])
                    
                    print (Score,'learningRate=',i,
                                 'n_estimators=',j,
                                 'max_depth=',k,
                                 'min_samples_split=',m)                
                    Dict[(i,j,k, m)] = Score
                    print "--- %s seconds --- " %(time.clock() - Start_Timer)
            
    A = min(Dict, key=Dict.get)
    print Dict[A]
    GTrue = GradientBoostingClassifier(n_estimators=A[1],
                                   learning_rate=A[0],
                                   max_depth = A[2],
                                   min_samples_split = A[3],
                                   verbose=1).fit(df_Xtrain, df_Ytrain)    
    df_Data.set_index(ID, inplace=True)
    B = GTrue.predict_proba(df_Data)
    df_result = pd.DataFrame(B)
    df_result.rename(columns={0:'proba0', 1:'target'}, inplace= True)
    df_Final = pd.concat([df_result, pd.DataFrame(df_Data.index)], axis =1)
    del df_Final['proba0']
    df_Final.set_index(ID, inplace=True)
    df_Final.to_csv('/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/result2.csv', sep=',')       
    L.append(A)
    L.append(Dict)
    L.append(df_Final)
    print "--- %s seconds --- " %(time.clock() - Start_Timer)
#   return L


Result = ModelGradientboosting()



#   
#    predictions = []
#    predictions = pd.DataFrame(predictions)
#    predictions['Gradient Boosting'] = GBest.predict(df_Xtest)
#




#    
#    df_Data.set_index(ID, inplace=True)
#    A = GBest.predict_proba(df_Data)
#    df_result = pd.DataFrame(A)
#    df_result.rename(columns={0:'proba0', 1:'target'}, inplace= True)
#    df_Final = pd.concat([df_result, pd.DataFrame(df_Data.index)], axis =1)
#    del df_Final['proba0']
#    df_Final.set_index(ID, inplace=True)
##   df_Final.to_csv('/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/result2.csv', sep=',')
#    
#    
##    caca= df_result[df_result['proba1']>0.3]
#    print predictions.head(50)
                          
    

    
    
 #   scores_predictions = pd.DataFrame()
"""  scores_predictions['Gradient Boosting'] = pd.Series([mean_squared_error(GBest.predict(df_Xtest), df_Ytest), r2_score(GBest.predict(df_Xtest), df_Ytest)], index=['MSE', 'R2'], name='Gradient Boosting')
    resultats_GB = pd.DataFrame({'Id':df_test.index, 'proba' : pd.DataFrame(np.expm1(GBest.predict(df_test)))[0]})
    
    return resultats_GB
"""
