# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:46:24 2019

@author: Christof Naumzik
"""

import pystan
from pystan import StanModel 
import stan_utility
import os
os.chdir('/local/home/cnaumzik/Ratings')
print('Load Data')
data = pystan.read_rdump('Data/Aggregation_data.R')
model_name = 'lgpm_mean'
print('Data loaded - compiling model ' + model_name)

model = StanModel('Code/Stan Code/' + model_name + '.stan')

print('Model compiled - Fitting  model')

fit = model.sampling(data = data, init = "0", iter = 2000, chains = 2, refresh = 10 , pars=['predictions','rng_ratings','log_lik','fitted_values','omega'])

print('Model fit successful - Saving fit now')

stan_utility.save_fit.stanfit_to_hdf5(fit = fit, file_name = "Data/model_fit_" + model_name)


