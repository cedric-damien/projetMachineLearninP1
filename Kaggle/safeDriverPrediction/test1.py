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

np.random.seed(0)
path_train = '/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/train.csv'
path_test = '/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/test.csv'

df_tot = pd.read_csv(path_train)
df_Data = pd.read_csv(path_test)
print df_tot[df_tot['id']==1063865]

def ModelGradientboosting(df = df_tot, sample = 10000, ID = 'id',target='target'):
    df_train=df.sample(n=sample,frac=None,random_state=0)
    df_test=df[~df[ID].isin(df_train[ID])]
    df_train.set_index(ID, inplace=True)
    df_test.set_index(ID, inplace=True)
   
    df_Ytrain = pd.DataFrame(df_train[target])
    df_Xtrain = df_train.drop(target, axis=1)

    df_Ytest = pd.DataFrame(df_test[target])
    df_Xtest = df_test.drop(target, axis=1)

    
    GBest = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0).fit(df_Xtrain, df_Ytrain)

    scores_mse = cross_val_score(GBest, df_Xtrain, df_Ytrain[target], cv=2, scoring="neg_mean_squared_error")
    scores_accuracy = cross_val_score(GBest, df_Xtrain, df_Ytrain[target], cv=2)


    print("For a Linear Regression,The Mean-Squared Error reaches : %.6f" % (-scores_mse.mean()))
    print("The corresponding R2 equals to : %.4f" % (scores_accuracy.mean()))
    
    
    print("For a GBR,The Mean-Squared Error reaches : %.6f" % (mean_squared_error(GBest.predict(df_Xtest), df_Ytest)))
    print("The corresponding R2 equals to : %.4f" % (r2_score(GBest.predict(df_Xtest), df_Ytest)))
    
    predictions = []
    predictions = pd.DataFrame(predictions)
    predictions['Gradient Boosting'] = GBest.predict(df_Xtest)





    
    df_Data.set_index(ID, inplace=True)
    A = GBest.predict_proba(df_Data)
    df_result = pd.DataFrame(A)
    df_result.rename(columns={0:'proba0', 1:'target'}, inplace= True)
    df_Final = pd.concat([df_result, pd.DataFrame(df_Data.index)], axis =1)
    del df_Final['proba0']
    df_Final.set_index(ID, inplace=True)
#    df_Final.to_csv('/Users/cedric/Documents/projetMachineLearninP1/Kaggle/safeDriverPrediction/result2.csv', sep=',')
    
    
#    caca= df_result[df_result['proba1']>0.3]
    print predictions.head(50)
                          
    

    
    
 #   scores_predictions = pd.DataFrame()
"""  scores_predictions['Gradient Boosting'] = pd.Series([mean_squared_error(GBest.predict(df_Xtest), df_Ytest), r2_score(GBest.predict(df_Xtest), df_Ytest)], index=['MSE', 'R2'], name='Gradient Boosting')
    resultats_GB = pd.DataFrame({'Id':df_test.index, 'proba' : pd.DataFrame(np.expm1(GBest.predict(df_test)))[0]})
    
    return resultats_GB
"""
