# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:47:40 2018

@author: arlan
"""

import numpy as np
from taqm import taqm
from scipy.stats import linregress
from beinf import beinf
import matplotlib.pyplot as plt
import os

# Change directory to where the data is stored and load data
os.chdir('Data')
X = np.load('MH.npy')   #load MH data
Y = np.load('OH.npy')   #load OH data
X_t = np.load('Raw_fcst.npy')   #load raw forecast
Y_t = 0.2 #made-up observation


# Time
tau_s = 1981    #start year
tau_f = 2012    #finish
tau = np.arange(tau_s,tau_f+1)  #array of years in hindcast record

t = 2011   #forecast year
tau_t = tau[tau!=t]   # remove the forecast year from tau and call it tau_t
 
#instantiate a taqm object
taqm = taqm()

# Get TAMH from MH
pval_x = linregress(tau_t,X.mean(axis=1))[3]  #check p-value for MH trend over tau_t                 
if pval_x<0.05:
    # if significant, then adjust MH for the trend to create TAMH
    X_ta = taqm.trend_adjust_2p(X,tau_t,t)
else:
    # else, set TAMH equal to MH (i.e. don't perform the trend adjustment) 
    X_ta = np.copy(X)

# Get TAOH from OH   
pval_y = linregress(tau_t,Y)[3]     #check p-value for OH trend over tau_t             
if pval_y<0.05:   
    # if significant, then adjust OH for the trend to create TAOH
    Y_ta = taqm.trend_adjust_2p(Y,tau_t,t) 
else:
    # else, set TAOH equal to OH (i.e. don't perform the trend adjustment) 
    Y_ta = np.copy(Y)
    
# Fit TAMH, TAOH, and X_t to the BEINF distribution
X_ta_params, Y_ta_params, X_t_params = taqm.fit_params(X_ta,Y_ta,X_t)   

# Calibrate forecast
trust_sharp_fcst = False

X_t_cal_params, X_t_cal = taqm.calibrate(X_ta_params, Y_ta_params, X_t_params,
                                                   X_ta, Y_ta, X_t,trust_sharp_fcst) 
print np.around(X_t_cal_params,4)
print np.around(X_t_cal,4)

# Get the individual parameters from the array containing the parameters
a_x_ta, b_x_ta, p_x_ta, q_x_ta = taqm.unpack_params(X_ta_params)
a_y_ta, b_y_ta, p_y_ta, q_y_ta = taqm.unpack_params(Y_ta_params)
a_x_t, b_x_t, p_x_t, q_x_t = taqm.unpack_params(X_t_params)
a_x_t_cal, b_x_t_cal, p_x_t_cal, q_x_t_cal = taqm.unpack_params(X_t_cal_params)

# freeze distribution objects
rv_x_ta = beinf(a_x_ta, b_x_ta, p_x_ta, q_x_ta) #TAMH                               
rv_y_ta = beinf(a_y_ta, b_y_ta, p_y_ta, q_y_ta) #TAOH
rv_x_t = beinf(a_x_t, b_x_t, p_x_t, q_x_t) #Raw forecast ensemble
rv_x_t_cal = beinf(a_x_t_cal, b_x_t_cal, p_x_t_cal, q_x_t_cal) #Calibrated forecast ensemble
    
x = np.linspace(0, 1, 1000)

# Evaluate cdf for the TAMH distribution at x
cdf_x_ta = beinf.cdf_eval(x,X_ta_params,X_ta)

# Evaluate cdf for the TAOH distribution at x
cdf_y_ta = beinf.cdf_eval(x,Y_ta_params,Y_ta)

# Evaluate cdf for the forecast distribution at x 
cdf_x_t = beinf.cdf_eval(x,X_t_params,X_t)

# Evaluate cdf for the calibrated forecast distribution at x
if trust_sharp_fcst==True and p_x_t==1:
    cdf_x_t_cal = beinf.cdf_eval(x,X_t_params,X_t)
else:
    if p_x_t==1.0:
        cdf_x_t_cal = beinf.cdf_eval(x,Y_ta_params,Y_ta)
    else:
        cdf_x_t_cal = beinf.cdf_eval(x,X_t_cal_params,X_t_cal)
        
        
fig = plt.figure()            
#Plot the cdf's for each of these distributions
ax1 = fig.add_subplot(1,2,1)
ax1.plot(x, cdf_x_ta, 'green', lw=1.5, label='TAMH')
ax1.plot(x, cdf_y_ta, 'orange', lw=1.5, label='TAOH')
ax1.legend(loc='lower right')
ax1.set_xlabel('Sea Ice Concentration')
ax1.set_ylabel('CDF')
ax1.set_ylim((-0.01, 1.01))
ax1.set_xlim((-0.01, 1.01))

ax2 = fig.add_subplot(1,2,2)
ax2.plot(x, cdf_x_t, 'b', lw=1.5, label='Raw fcst')
ax2.plot(x, cdf_x_t_cal, 'r', lw=1.5, label='TAQM fcst')
ax2.legend(loc='lower right')
ax2.set_xlabel('Sea Ice Concentration')
ax2.set_ylabel('CDF')
ax2.set_ylim((-0.01, 1.01))
ax2.set_xlim((-0.01, 1.01))

fig.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.9,
                    wspace=0.25)
 
#heaviside function for obs                                
cdf_obs = np.zeros(len(x))
cdf_obs[Y_t*np.ones(len(x))<=x] = 1.0
 
crps_x_t = np.trapz((cdf_x_t - cdf_obs)**2.,x)
crps_x_t_cal = np.trapz((cdf_x_t_cal - cdf_obs)**2.,x)

print "CRPS: raw forecast: ", crps_x_t                            
print "CRPS: calibrated forecast: ", crps_x_t_cal
