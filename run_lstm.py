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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from sklearn.preprocessing import MinMaxScaler
import os

os.chdir('/local/home/cnaumzik/Ratings')
n_neurons = 25
n_epochs = 50
n_steps = np.int(sys.argv[1])
columns = ['MAE', 'RMSE', 'JS-Div', 'Wasserstein']

results_tbl = pd.DataFrame(columns = columns)
def split_sequence(sequence, Data, n_steps):
    X, y, D = list(), list(),list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        seq_d = Data[i:end_ix,:]
        X.append(seq_x)
        y.append(seq_y)
        D.append(seq_d)
    return np.array(X), np.array(y),np.array(D)

def fit_lstm(one_hot_input,one_hot_labels, batch_size, n_epochs, n_neurons,n_steps,n_feat):
    model = Sequential()
    model.add(LSTM(n_neurons , 
                   batch_input_shape = (batch_size,n_steps, n_feat), 
                   dropout = 0.5, 
                   return_sequences = False, 
                   stateful=True))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
    for i in range(n_epochs):
        model.fit(one_hot_input, one_hot_labels,batch_size=batch_size ,epochs=1, verbose=0, shuffle=False)
        model.reset_states()
    return model

data = pystan.read_rdump('Data/Aggregation_data_short_log.R')
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
    sequence = Y[idx:(idx+N[i])]
    x_t,y_t,D_t = split_sequence(sequence,COV[idx:(idx+N[i]),:],n_steps)
    y_t = K.utils.to_categorical(y_t-1, num_classes=5)
    x_t = K.utils.to_categorical(x_t-1,num_classes=5)
    D_t = np.append(x_t,D_t,axis=2)
    #Simple LSTM Rating sequence only
    m = fit_lstm(one_hot_input = x_t, one_hot_labels = y_t , 
             batch_size = 1, n_epochs = n_epochs, n_neurons = n_neurons, n_steps = n_steps, n_feat = 5)
    x_test = sequence[(N[i]-n_steps):N[i]]
    x_test = K.utils.to_categorical(x_test-1,num_classes=5)
    x_test = x_test.reshape((1,x_test.shape[0],x_test.shape[1]))
    y_prop = m.predict(x_test,batch_size = 1).reshape(-1)
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
    m = fit_lstm(one_hot_input = D_t, one_hot_labels = y_t , 
             batch_size = 1, n_epochs = n_epochs, n_neurons = n_neurons, n_steps = n_steps, n_feat = D_t.shape[2])
    D_test = COV[idx:(idx+N[i])]
    D_test = D_test[(N[i]-n_steps):N[i],:]
    D_test = D_test.reshape((1,D_test.shape[0],D_test.shape[1]))
    
    x_test = sequence[(N[i]-n_steps):N[i]]
    x_test = K.utils.to_categorical(x_test-1,num_classes=5)
    x_test = x_test.reshape((1,x_test.shape[0],x_test.shape[1]))
    D_test = np.append(x_test,D_test,axis=2)
    y_prop = m.predict(D_test,batch_size = 1).reshape(-1)
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
                 'Wasserstein':np.mean(wasserstein),
                 'Steps' : n_steps},name = 'Basic LSTM')
results_tbl = results_tbl.append(row)
row = pd.Series({'MAE':np.mean(error_full), 
                 'RMSE':np.sqrt(np.mean(np.power(error_full,2))),
                 'JS-Div':np.mean(js_full),
                 'Wasserstein':np.mean(wasserstein_full),
                 'Steps' : n_steps},name = 'Full variable LSTM')
results_tbl = results_tbl.append(row)    
with open('lstm_bidirectional_results_tbl_' + str(n_steps) + '.tex','w') as lf:
    lf.write(results_tbl.to_latex(float_format=lambda x: '%.4f' % x))    