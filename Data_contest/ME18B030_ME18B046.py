#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer
from sklearn.utils import resample
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectKBest, chi2
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

VALIDATION = False

train1_df = pd.read_csv(f'Dataset_1_Training.csv',index_col=0).T
train2_df = pd.read_csv(f'Dataset_2_Training.csv',index_col=0).T

test1_df = pd.read_csv(f'Dataset_1_Testing.csv',index_col=0).T
test2_df = pd.read_csv(f'Dataset_2_Testing.csv',index_col=0).T

train1 = train1_df
train2 = train2_df
test1 = test1_df
test2 = test2_df

if(VALIDATION):
    #Shuffling the dataset
    valSize = int(train1.shape[0]*0.7)
    train1 = train1.sample(frac=1).reset_index(drop=True)
    test1 = train1[valSize:]
    train1 = train1[:valSize]
    
    valSize = int(train2.shape[0]*0.7)
    train2 = train2.sample(frac=1).reset_index(drop=True)
    test2 = train2[valSize:]
    train2 = train2[:valSize]

print("Loaded Dataset_1_Training ",train1.shape)
print("Loaded Dataset_1_Testing ", test1.shape)
print("Loaded Dataset_2_Training ",train2.shape)
print("Loaded Dataset_2_Testing ", test2.shape)

test_cols1 = ['CO: 1','CO: 2']
test_cols2 = ['CO: 3','CO: 4','CO: 5','CO: 6']
test_cols = ['CO: 1','CO: 2','CO: 3','CO: 4','CO: 5','CO: 6']


#For RF
random_grid = {'model__n_estimators': [10, 25, 50, 100, 250, 500],
               'model__max_depth': [3, 5, 10, 25, None],
               'select__k': [500,1000,1500,2000,2500]}

# #For SVC
# random_grid = {'C': [0.25, 0.5, 1, 2.5, 5],
#                'kernel': ['linear', 'poly', 'rbf'],
#                'gamma': ['scale','auto']}



default_param = random_grid
default_model = RandomForestClassifier(random_state=0)
def pred(train,test,col,model=default_model,params=default_param):
    X = train.drop(test_cols, axis=1, errors='ignore')
    y = np.array(train[[col]])
    y = y.reshape(y.shape[0],)
    Xtest = test.drop(test_cols, axis=1, errors='ignore')

    if(VALIDATION): 
        ytest = test[[col]]
        ytrue.extend(list(ytest[col]))
        
    pipe = Pipeline(steps=[("scale", StandardScaler()),("select", SelectKBest()), ("model", model)])
    
    model = GridSearchCV(pipe,params,verbose=0,n_jobs=4,cv=ShuffleSplit(n_splits=1,test_size=0.3,random_state=0))
    model.fit(X,y)
    best_params.append(model.best_params_)
    # print(model.best_params_)
    
    # print(np.unique(model.predict(Xtest),return_counts=True))
      
    ypred.extend(model.predict(Xtest))   

    if(VALIDATION): print(col,model.score(Xtest,ytest),accuracy_score(ytest,model.predict(Xtest)))
    else: print("Trained ",col)
        
    return pd.DataFrame(model.cv_results_).sort_values(by=['rank_test_score']).head(10)

ypred = []
ytrue = []
best_params = []

random_grid = {'model__n_estimators': [25],
               'model__max_depth': [3],
               'select__k': [2500]}
pred(train1,test1,'CO: 1',model=RandomForestClassifier(random_state=0),params=random_grid)


random_grid = {'model__n_estimators': [50],
               'model__max_depth': [3],
               'select__k': [500]}
pred(train1,test1,'CO: 2',params=random_grid)


random_grid = {'model__n_estimators': [200],
               'model__max_depth': [5],
               'select__k': [2500]}
pred(train2,test2,'CO: 3',params=random_grid)


random_grid = {'model__n_estimators': [25],
               'model__max_depth': [5],
               'select__k': [1500]}
pred(train2,test2,'CO: 4',params=random_grid)


random_grid = {'model__n_estimators': [50],
               'model__max_depth': [5],
               'select__k': [500]}
pred(train2,test2,'CO: 5',params=random_grid)


svc_grid = {'model__kernel': ['poly'],
            'model__C': [0.25],
            'select__k': [1500]}
pred(train2,test2,'CO: 6',SVC(random_state=0),svc_grid)


if(VALIDATION):
    print(matthews_corrcoef(ytrue,ypred))
else:
    ypred = np.array(ypred)
    submission = pd.DataFrame(ypred,columns=['Predicted'])
    submission.index.name = 'Id'
    submission.Predicted = np.array(ypred,dtype=int)
    submission.to_csv('ME18B030_ME18B046.csv',index=True)
    print("Generated ME18B030_ME18B046.csv with shape ",submission.shape)




