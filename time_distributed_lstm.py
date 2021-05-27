# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:43:31 2019

@author: Christof Naumzik
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 06:29:43 2019

@author: Christof Naumzik
"""
import sys
import pystan
import numpy as np
import keras as K
from keras import backend as backend 
import pandas as pd
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, LSTM, Dropout

import os

os.chdir('/local/home/cnaumzik/Ratings')
n_neurons = np.int(sys.argv[1])
n_epochs = 20
n_R = 5
columns = ['MAE', 'RMSE', 'JS-Div', 'Wasserstein']

results_tbl = pd.DataFrame(columns = columns)
class MyBatchGenerator(K.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *X[index].shape))
        yb = np.empty((self.batch_size, *y[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = X[index]
            yb[s] = y[index]
        return Xb, yb 
    
def create_samples(sequence,min_size):
    X = []
    y = []
    i = min_size
    while i < sequence.shape[0]:
        X.append(sequence[0:i,:])
        y.append(sequence[i,:].reshape(1,n_R))
        i = i +1
    return np.array(X),np.array(y)
def create_samples_full(sequence,D,min_size):
    X = []
    y = []
    i = min_size
    while i < sequence.shape[0]:
        X.append(np.append(sequence[0:i,:],D[0:i,:],axis=1))
        y.append(sequence[i,:].reshape(1,n_R))
        i = i +1
    return np.array(X),np.array(y)

def init_model(n_neurons,n_feat):
    model = Sequential()
    model.add(LSTM(n_neurons , 
                   input_shape = (None, n_feat),
                   return_sequences = True))
    model.add(Dropout(rate = 0.25))
    model.add(TimeDistributed(Dense(n_R, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
    return model

data = pystan.read_rdump('Data/Aggregation_data_short_log.R')
#data = pystan.read_rdump('Code/Rating Aggregation/Aggregation_data_short_log.R')
Y = data['Y']
N = data['N']
X = data['X']
Xtest = data['X_test']
Ytest = data['Y_test']
TEST_NUM = np.int(data['TEST_NUM'])
NUM_BUSINESS = np.int(data['NUM_BUSINESS'])
Q = data["Q_COV"]
R = data["R_COV"]
COV = np.matmul(Q,R)
Q = data["Q_MEAN"]
R = data["R_MEAN"]
COV = np.append(COV,np.matmul(Q,R),axis=1)
#MinMax Scaling
scaler = MinMaxScaler(feature_range=(0,1))
for i in range(COV.shape[1]):
    COV[:,i] = scaler.fit_transform(COV[:,i].reshape(-1,1)).reshape(-1)
#Data preparation
idx = 0
error = []
js = []
wasserstein = []
error_full = []
js_full = []
wasserstein_full = []
for i in range(NUM_BUSINESS):
    print(i)     
    sequence = K.utils.to_categorical(Y[idx:(idx+N[i])]-1, num_classes=n_R)
    X,y = create_samples(sequence,5)
    TrainData = MyBatchGenerator(X=X,y=y)
    m = init_model(n_neurons = n_neurons, 
                   n_feat = n_R)
    m = m.fit_generator(TrainData,epochs=n_epochs,verbose=1)
    #Simple LSTM Rating sequence only
    y_prop = m.model.predict(sequence.reshape(1,-1,n_R))[:,N[i]-1,:].reshape(-1)
    y_hat = np.matmul(y_prop,np.arange(1,6))
    lstm_samples = np.random.choice(np.arange(1,6),size=2000,replace=True,p=y_prop)
    error.append(abs(Ytest[(i*TEST_NUM):((i+1)*TEST_NUM)].mean()-y_hat).reshape(-1))
    _,q  = np.unique(np.append(Ytest[(i*TEST_NUM):((i+1)*TEST_NUM)], np.arange(1,6)), return_counts=True)
    q = q - 1 
    q = q /np.sum(q)
    test_samples = np.random.choice(np.arange(1,6),size=2000,replace=True,p=q)
    js.append(np.power(distance.jensenshannon(y_prop,q),2))
    wasserstein.append(wasserstein_distance(test_samples,lstm_samples))
    backend.clear_session()
    #LSTM with all rating variables
    X,y = create_samples_full(sequence,COV[idx:(idx+N[i])],5)
    TrainData = MyBatchGenerator(X=X,y=y)
    m = init_model(n_neurons = n_neurons, 
                   n_feat = n_R+COV.shape[1])
    m = m.fit_generator(TrainData,epochs=n_epochs,verbose=1)
    y_prop = m.model.predict(np.append(sequence,COV[idx:(idx+N[i])],axis=1).reshape(1,-1,n_R+COV.shape[1]))[:,N[i]-1,:].reshape(-1)
    y_hat = np.matmul(y_prop,np.arange(1,6))
    lstm_samples = np.random.choice(np.arange(1,6),size=2000,replace=True,p=y_prop)
    error_full.append(abs(Ytest[(i*TEST_NUM):((i+1)*TEST_NUM)].mean()-y_hat).reshape(-1))
    _,q  = np.unique(np.append(Ytest[(i*TEST_NUM):((i+1)*TEST_NUM)], np.arange(1,6)), return_counts=True)
    q = q - 1 
    q = q /np.sum(q)
    test_samples = np.random.choice(np.arange(1,6),size=2000,replace=True,p=q)
    js_full.append(np.power(distance.jensenshannon(y_prop,q),2))
    wasserstein_full.append(wasserstein_distance(test_samples,lstm_samples))
    backend.clear_session()
    idx = idx + N[i]


row = pd.Series({'MAE':np.mean(error), 
                 'RMSE':np.sqrt(np.mean(np.power(error,2))),
                 'JS-Div':np.mean(js),
                 'Wasserstein':np.mean(wasserstein)},name = 'Basic LSTM')
results_tbl = results_tbl.append(row)
row = pd.Series({'MAE':np.mean(error_full), 
                 'RMSE':np.sqrt(np.mean(np.power(error_full,2))),
                 'JS-Div':np.mean(js_full),
                 'Wasserstein':np.mean(wasserstein_full)},name = 'Full variable LSTM')
results_tbl = results_tbl.append(row)    
with open('lstm_results_tbl_' + str(n_neurons) + '.tex','w') as lf:
    lf.write(results_tbl.to_latex(float_format=lambda x: '%.4f' % x))    