# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:46:30 2019

@author: Christof Naumzik
"""

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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import timeit

from gpflow.training import  AdamOptimizer
plt.rcParams['figure.figsize'] = (12, 6)

columns = ['MAE', 'RMSE', 'JS-Div', 'Wasserstein']
models = ['GP', 'Sample Mean']
results = pd.DataFrame(columns = columns)

os.chdir('/local/home/cnaumzik/Ratings')  
TARGET_CITY = "Phoenix"
MIN_REVIEWS = 200


MIN_YEAR = 2010
TEST_NUM = 20
business = pd.read_csv(r'Data/yelp_business.csv')
##Only business from target city
business = business[business["city"] == TARGET_CITY ]
business = business[business["categories"].str.contains('Restaurant')]
business=business.drop("categories",axis=1)
reviews = pd.read_csv(r'Data/yelp_review.csv')
print(len(reviews))
reviews = reviews[reviews["business_id"].isin(business["business_id"])]
##get year&number of reviews
year = []
freq = []
for k in np.nditer(np.arange(0,len(business))):
    it = np.int(k)
    Data = reviews.loc[reviews["business_id"].isin(business["business_id"][it:(it+1)]),["date"]]
    Data["date"] = pd.to_datetime(Data["date"])
    Data = Data.set_index('date',drop=False)
    Data = Data.sort_index()  
    year.append(int(str(Data.iloc[0,0])[0:4]))
    freq.append(len(Data['date']))

year = np.array(year)
freq = np.array(freq)
business['year'] = pd.Series(year.reshape(-1),index = business.index)
business['freq'] = pd.Series(freq.reshape(-1),index = business.index)
business = business[business["year"]>=MIN_YEAR]
business = business[business["freq"]>=MIN_REVIEWS]

##Select only reviews for the selected businesses
reviews = reviews[reviews["business_id"].isin(business["business_id"])]



def init():
    num_inducing = np.int(X.shape[0]*0.8)
    L = np.random.choice(np.arange(0,X.shape[0]),replace = False,size=num_inducing)
    Z = X[L,:]
    mu = (minDiff + maxDiff + 1.0)/2.0
    sig = (maxDiff - minDiff)/6.0
    scale = sig/np.sqrt(mu)
    shape = np.power(mu,1.5)/sig
    bin_edges = np.array(np.arange(4), dtype=float)
    bin_edges = bin_edges - bin_edges.mean()
    likeli_hood = gpflow.likelihoods.Ordinal(bin_edges)
    # create SVGP model as usual and optimize
    m = gpflow.models.SVGP(X = X, Y = Y, Z = Z,
                          likelihood = likeli_hood, 
                          kern = gpflow.kernels.Matern12(1))
    m.feature.trainable = True
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
length_list = []
id_list = []
#GP lists
error = []
js = []
wasserstein = []
#Sample Mean
sm_error = []
sm_js = []
sm_wasserstein = []
scaler = StandardScaler()
run_time = []
for k in range(len(business)):
    print(k)
    it = np.int(k)
    id_list.append(business["business_id"][it:(it+1)])
    Data = reviews.loc[reviews["business_id"].isin(business["business_id"][it:(it+1)]),["date","stars"]]
    Data["date"] = pd.to_datetime(Data["date"])
    Data = Data.set_index('date',drop=False)
    Data = Data.sort_index() 
    N = len(Data) - TEST_NUM
    length_list.append(len(Data))
    Y = Data["stars"].values.reshape(-1,1)
    Y_test = Y[N:(N+TEST_NUM)]
    Y = Y[0:N] - 1
    X = Data["date"]-Data["date"][0]
    X = X.dt.days.values.reshape(-1,1).astype(np.float64)
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    minDiff = 1.0/scaler.scale_
    maxDiff = X.max() - X.min()
    X = X + np.random.normal(0.0,0.001,N + TEST_NUM).reshape(-1,1)
    X_test = X[N:(N + TEST_NUM)]
    X = X[0:N]
    start = timeit.default_timer()
    m = test_train(iterations = 5000)
    stop = timeit.default_timer()
    print(stop - start)
    run_time.append(stop - start)
    mu, _ = m.predict_y(X_test)
    mu = 1 + mu 
    #Calculate GP performance
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
    #Calculate Sample mean performance
    _,p  = unique, counts = np.unique(np.append(Y, np.arange(0,5)), return_counts=True)
    p = p - 1
    p = p /np.sum(p)
    train_samples = np.random.choice(np.arange(1,6),size=1000,replace=True,p=p)

    Y = 1 + Y
    sm_js.append(np.power(distance.jensenshannon(p,q),2))
    sm_wasserstein.append(wasserstein_distance(test_samples,train_samples))
    sm_error.append(abs(Y_test.mean()-Y.mean()))
    
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    gpflow.reset_default_session(graph=graph)


error_df = pd.DataFrame({'ID': id_list,'Length': length_list, 'Mean': sm_error, 'LGPM': error})

error_df.to_csv("error_comparison.csv")


row = pd.Series({'MAE':np.mean(error), 
                 'RMSE':np.sqrt(np.mean(np.power(error,2))),
                 'JS-Div':np.mean(js),
                 'Wasserstein':np.mean(wasserstein)},name="GP")
results = results.append(row)
row = pd.Series({'MAE':np.mean(sm_error), 
                 'RMSE':np.sqrt(np.mean(np.power(sm_error,2))),
                 'JS-Div':np.mean(sm_js),
                 'Wasserstein':np.mean(sm_wasserstein)},name="Sample Mean")
results = results.append(row)
business['run_time'] = pd.Series(np.array(run_time).reshape(-1),index = business.index)
business.to_csv("business_long_seq.csv")
with open('gpflow_long_seq_results_tbl.tex','w') as lf:
    lf.write(results.to_latex(float_format=lambda x: '%.3f' % x))        
