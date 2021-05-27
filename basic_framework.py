# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:24 2019

@author: Christof Naumzik
"""
import numpy as np
import gpflow
import pandas as pd
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import tensorflow as tf
import pystan
import os
from gpflow.training import  AdamOptimizer
plt.rcParams['figure.figsize'] = (12, 6)


os.chdir('/local/home/cnaumzik/Ratings')  

def init():
    mu = (minDiff + maxDiff + 1.0)/2.0
    sig = (maxDiff - minDiff)/6.0
    scale = sig/np.sqrt(mu)
    shape = np.power(mu,1.5)/sig
    bin_edges = np.array(np.arange(4), dtype=float)
    bin_edges = bin_edges - bin_edges.mean()
    likeli_hood = gpflow.likelihoods.Ordinal(bin_edges)
    # create SVGP model as usual and optimize
    m = gpflow.models.VGP(X = X, Y = Y, 
                          likelihood = likeli_hood, 
                          kern = gpflow.kernels.Matern12(1))
    m.likelihood.trainable = True
    m.likelihood.variance = 0.01
    m.kern.lengthscales.prior  = gpflow.priors.Gamma(shape,scale)
    return m

def test_train(iterations):   
    with gpflow.defer_build():
        m = init()
       
    tf.local_variables_initializer()
    tf.global_variables_initializer()

    tf_session = m.enquire_session()
    m.compile( tf_session )
    op_adam = AdamOptimizer(0.01).make_optimize_tensor(m)
    for it in range(iterations):           
        tf_session.run(op_adam)
        if it % 100 == 0:
            likelihood = tf_session.run(m.likelihood_tensor)
            print('{}, ELBO={:.4f}'.format(it, likelihood))
    
    m.anchor(tf_session)
    
    return m


data = pystan.read_rdump('Data/Aggregation_data.R')
N_full = data['N']
X_full = data['X']
X_test_full = data['X_test']
Y_full = data['Y']
Y_test_full = data['Y_test']
minDiff_full = data['minDiff']    
maxDiff_full = data['maxDiff']
M = data['TEST_NUM']
NUM_BUSINESS = data['NUM_BUSINESS']

error = []
js = []
wasserstein = []
idx = 0
for k in range(NUM_BUSINESS):
     N = N_full[k]
     minDiff = minDiff_full[k]
     maxDiff = maxDiff_full[k]
     X_test = X_test_full[k*M:(k+1)*M].reshape(-1,1)
     Y_test = Y_test_full[k*M:(k+1)*M].reshape(-1,1)
     Y = Y_full[idx:(idx+N)].reshape(-1,1)
     Y = Y - 1
     X = X_full[idx:(idx+N)].reshape(-1,1)
     m = test_train(iterations = 2500)   
     mu, _ = m.predict_y(X_test)
     mu = 1 + mu 
     idx = idx + N
     y_prop = np.zeros((5,))
     for r in np.arange(0,5):
         y_prop[r] = np.exp(m.predict_density(X_test, np.ones_like(X_test) * r)).mean()
     y_prop = y_prop/sum(y_prop)    
     gp_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=y_prop)
     error.append(abs(Y_test.mean() - mu.mean()).reshape(-1))
     _,q  = unique, counts = np.unique(np.append(Y_test, np.arange(1,6)), return_counts=True)
     q = q - 1 
     q = q /np.sum(q)
     test_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=q)
     js.append(np.power(distance.jensenshannon(y_prop,q),2))
     wasserstein.append(wasserstein_distance(test_samples,gp_samples))
     tf.reset_default_graph()
     graph = tf.get_default_graph()
     gpflow.reset_default_session(graph=graph)

row = pd.Series({'MAE':np.mean(error), 
                 'RMSE':np.sqrt(np.mean(np.power(error,2))),
                 'JS-Div':np.mean(js),
                 'Wasserstein':np.mean(wasserstein)})
    
with open('gpflow_results_tbl.tex','w') as lf:
    lf.write(row.to_latex(float_format=lambda x: '%.3f' % x))        
