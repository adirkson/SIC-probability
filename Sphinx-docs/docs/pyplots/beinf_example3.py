# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:47:40 2018

@author: arlan
"""

from beinf import beinf   
import matplotlib.pyplot as plt
import numpy as np

a, b, p, q = 2.3, 6.7, 0.4, 0.0
rv = beinf(a, b, p, q)

x_l = 0.15
sip = 1.0 - rv.cdf(x_l)

x = np.linspace(0.0, 1.0, 1000)

X = np.load('sample.npy')
#Fit these random variates to the beinf distribution
a_f, b_f, p_f, q_f = beinf.fit(X)
rv_f = beinf(a_f, b_f, p_f, q_f)

#Plot the exact pdf, histogram for the random sample, and the fitted pdf
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.plot(x, rv.pdf(x), 'r-',label='exact pdf', lw=1.5)
ax1.hist(X,normed=True,label='sample hist',histtype='stepfilled')
ax1.plot(x, rv_f.pdf(x), 'g-',label='fitted pdf', lw=1.5)
ax1.legend(loc='upper right')
ax1.plot(0.0, p*(1-q), 'ro', ms=6)
ax1.plot(1.0, p*q, 'ro', ms=6)
ax1.plot(0.0, p_f*(1-q_f), 'go', ms=6)
ax1.plot(1.0, p_f*q_f, 'go', ms=6)
ax1.set_xlim((-0.01,1.01))
ax1.set_xlabel('x',fontsize=12)
ax1.set_ylabel('Probability Density',fontsize=12)
ax1.set_title('Probability \n Density Function',fontsize=14)

#Plot the exact cdf, ecdf for the random sample, and the fitted cdf
ax2 = fig.add_subplot(1,2,2)
ax2.plot(x, rv.cdf(x), 'r',label='exact cdf', lw=1.5)
ax2.plot(x, beinf.ecdf(x, X), 'b',label='sample ecdf')
ax2.plot(x, rv_f.cdf(x), 'g',label='fitted cdf', lw=1.5)
ax2.legend(loc='lower right')
ax2.set_xlim((-0.01,1.01))
ax2.set_ylim((0,1))   
ax2.set_xlabel('x',fontsize=12)
ax2.set_ylabel(r'$P(X\leq x)$',fontsize=12)
ax2.set_title('Cumulative \n Distribution Function',fontsize=14)

fig.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.9,
                    wspace=0.25)