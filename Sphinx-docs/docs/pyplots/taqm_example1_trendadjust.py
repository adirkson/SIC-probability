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
X = np.load('MH_ex1.npy')   #load MH data
Y = np.load('OH_ex1.npy')   #load OH data
X_t = np.load('Raw_fcst_ex1.npy')   #load raw forecast
Y_t = 0.2 #made-up observation


# Time
tau_s = 1981    #start year
tau_f = 2017    #finish
tau = np.arange(tau_s,tau_f+1)  #array of years in hindcast record

t = 2012   #forecast year
tau_t = tau[tau<t]   # remove the forecast year from tau and call it tau_t
 
#instantiate a taqm object
taqm = taqm()

# Get TAMH from MH
pval_x = linregress(tau_t,X.mean(axis=1))[3]  #check p-value for MH trend over tau_t                 
if pval_x<0.05:
    # if significant, then adjust MH for the trend to create TAMH
    X_ta = taqm.trend_adjust_1p(X,tau_t,t)
else:
    # else, set TAMH equal to MH (i.e. don't perform the trend adjustment) 
    X_ta = np.copy(X)

# Get TAOH from OH   
pval_y = linregress(tau_t,Y)[3]     #check p-value for OH trend over tau_t             
if pval_y<0.05:   
    # if significant, then adjust OH for the trend to create TAOH
    Y_ta = taqm.trend_adjust_1p(Y,tau_t,t) 
else:
    # else, set TAOH equal to OH (i.e. don't perform the trend adjustment) 
    Y_ta = np.copy(Y)

fig = plt.figure()            
#Plot the cdf's for each of these distributions
ax1 = fig.add_subplot(2,1,1)    
ax1.plot(tau_t,X.mean(axis=1),'k',lw=2,label='MH')  
ax1.plot(tau_t,X_ta.mean(axis=1),'green',lw=2,ls='-',label='TAMH') 
ax1.legend(loc='lower left')


ax1.fill_between(tau_t,X.min(axis=1),
                 X.max(axis=1),color='k',alpha=0.2)  
ax1.fill_between(tau_t,X_ta.min(axis=1),
                 X_ta.max(axis=1),color='green',alpha=0.2) 
                 
ax1.set_ylim((-0.02,1.02))
ax1.set_ylabel('Sea Ice Concentration')

                 
ax2 = fig.add_subplot(2,1,2)    
ax2.plot(tau_t,Y,'k',lw=2,label='OH')  
ax2.plot(tau_t,Y_ta,'orange',lw=2,ls='-',label='TAOH') 
ax2.legend(loc='lower left')

ax2.set_ylim((-0.02,1.02))
ax2.set_ylabel('Sea Ice Concentration')

fig.subplots_adjust(bottom=0.1,right=0.99,left=0.1,top=0.99)