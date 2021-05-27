# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:18:45 2019

@author: SpAdmin
"""


import feather
import pystan
import adapted_fds
import spectral_meets_EM
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from pystan import StanModel
TEST_NUM = 10
columns = ['Hyper','MAE', 'RMSE', 'JS-Div', 'Wasserstein']
results = pd.DataFrame(columns = columns)


data_train = feather.read_dataframe('../../Data/crowdscourcing_data_train.feather')
data_test = feather.read_dataframe('../../Data/crowdscourcing_data_test.feather')
data_train = data_train[["RES_ID","USER_ID","stars"]]
data_train = data_train.astype({"stars":int})
data = pystan.read_rdump('Aggregation_data_short_log.R')
Ytest = data['Y_test']
Y = data['Y']
N = data['N']
d = {}
for x, y, z in zip(data_train.RES_ID, data_train.USER_ID, data_train.stars):
    d.setdefault(x,{})[y] = [z]

error = []
js = []
wasserstein = []
idx = 0
for i in range(N.shape[0]):
    sample_mean = Y[idx:(idx+N[i])].mean()
    Y_test = data_test.stars[(i*TEST_NUM):((i+1)*TEST_NUM)]
    error.append(abs(Y_test.mean()-sample_mean).reshape(-1))    
    _,q  = unique, counts = np.unique(np.append(Y_test, np.arange(1,6)), return_counts=True)
    q = q - 1
    q = q /np.sum(q)
    test_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=q)
    _,p  = unique, counts = np.unique(np.append(Y[idx:(idx+N[i])], np.arange(1,6)), return_counts=True)
    p = p - 1
    p = p /np.sum(p)
    train_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=p)
    js.append(np.power(distance.jensenshannon(p,q),2))
    wasserstein.append(wasserstein_distance(test_samples,train_samples))
    idx = idx + N[i]

row = pd.Series({'Hyper':"None",'MAE':np.mean(error), 
                 'RMSE':np.sqrt(np.mean(np.power(error,2))),
                 'JS-Div':np.mean(js),
                 'Wasserstein':np.mean(wasserstein)}, 
    name="Sample mean")

results = results.append(row)

#Fast DS implementation
algos = ['MV','DS','FDS','H']

for a in algos:
    print(a)
    DS_Q = adapted_fds.crowdsourcing(responses=d,algo=a)
    if a == 'MV':
        print(DS_Q)
    error = []
    js = []
    wasserstein = []
    idx = 0
    for i in range(DS_Q.shape[0]):
        Y_test = data_test.stars[(i*TEST_NUM):((i+1)*TEST_NUM)]
        y_hat = np.matmul(DS_Q[i,:],np.arange(1,6))
        error.append(abs(Y_test.mean()-y_hat).reshape(-1))
        _,q  = unique, counts = np.unique(np.append(Y_test, np.arange(1,6)), return_counts=True)
        q = q - 1
        q = q /np.sum(q)
        test_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=q)
        p = DS_Q[i,:]
        train_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=p)
        js.append(np.power(distance.jensenshannon(p,q),2))
        wasserstein.append(wasserstein_distance(test_samples,train_samples))
        idx = idx + N[i]
    row = pd.Series({'Hyper':a,'MAE':np.mean(error), 
                 'RMSE':np.sqrt(np.mean(np.power(error,2))),
                 'JS-Div':np.mean(js),
                 'Wasserstein':np.mean(wasserstein)},name="Fast Dawie-Skene")
    results = results.append(row)    
        

#Implementation of Bayesian Crowdsourcing models: Including Hierarchical Dawid-Skene and Logistic Random Effects model
algos = ['HDS-NC','LRE-NC',"ID-NC"]
stan_data = dict(J = data_train["USER_ID"].max(), 
            K = 5, 
            N = data_train.shape[0],
            I = data_train["RES_ID"].max(), 
            ii = np.array(data_train["RES_ID"],dtype="int32"), 
            jj = np.array(data_train["USER_ID"],dtype="int32"),
            y = np.array(data_train["stars"],dtype="int32"))
for a in algos:
    print(a)
    sm = StanModel(a+'.stan')
    op = sm.optimizing(data=stan_data,verbose=True)
    print("Opt run successfully")
    Q_Z = op['q_z']
    error = []
    js = []
    wasserstein = []
    idx = 0
    for i in range(Q_Z.shape[0]):
        Y_test = data_test.stars[(i*TEST_NUM):((i+1)*TEST_NUM)]
        y_hat = np.matmul(Q_Z[i,:],np.arange(1,6))
        error.append(abs(Y_test.mean()-y_hat).reshape(-1))
        _,q  = unique, counts = np.unique(np.append(Y_test, np.arange(1,6)), return_counts=True)
        q = q - 1
        q = q /np.sum(q)
        test_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=q)
        p = Q_Z[i,:]
        train_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=p)
        js.append(np.power(distance.jensenshannon(p,q),2))
        wasserstein.append(wasserstein_distance(test_samples,train_samples))
        idx = idx + N[i]
    row = pd.Series({'Hyper':a,'MAE':np.mean(error), 
                 'RMSE':np.sqrt(np.mean(np.power(error,2))),
                 'JS-Div':np.mean(js),
                 'Wasserstein':np.mean(wasserstein)},name="Paun et.al")
    results = results.append(row) 

    


_,Spectral_Q = spectral_meets_EM.spectral(A = data_train,B=data_test,N=N,algo="Spec",init='MV',N_iter=10,mode=0)
spectral_error = []
idx = 0
for i in range(Spectral_Q.shape[0]):
    Y_test = data_test.stars[(i*TEST_NUM):((i+1)*TEST_NUM)]
    y_hat = np.matmul(Spectral_Q[i,:],np.arange(1,6))
    spectral_error.append(abs(Y_test.mean()-y_hat).reshape(-1))    
    idx = idx + N[i]

np.mean(spectral_error)
np.sqrt(np.mean(np.power(spectral_error,2)))
