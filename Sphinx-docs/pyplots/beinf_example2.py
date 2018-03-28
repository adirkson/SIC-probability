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

nsamples = 20
X = rv.rvs(nsamples) # draw random sample
np.save('sample',X)
#Plot the exact pdf, histogram for the random sample, and the fitted pdf

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.plot(x, rv.pdf(x), 'r-',label='exact pdf', lw=1.5)
ax1.hist(X,normed=True,label='hist',histtype='stepfilled')
ax1.legend(loc='upper right')
# plot probability masses at 0 and 1 as circles
ax1.plot(0.0, p*(1-q), 'ro', ms=6)
ax1.plot(1.0, p*q, 'ro', ms=6)
ax1.set_xlim((-0.01,1.01))
ax1.set_xlabel('x',fontsize=12)
ax1.set_ylabel('Probability Density',fontsize=12)
ax1.set_title('Probability \n Density Function',fontsize=14)

#Plot the exact cdf, ecdf for the random sample, and the fitted cdf

ax2 = fig.add_subplot(1,2,2)
ax2.plot(x, rv.cdf(x), 'r',label='exact cdf', lw=1.5)
ax2.plot(x, beinf.ecdf(x, X), 'b',label='ecdf')
ax2.legend(loc='lower right')
ax2.set_xlim((-0.01,1.01))
ax2.set_ylim((0,1))   
ax2.set_xlabel('x',fontsize=12)
ax2.set_ylabel(r'$P(X\leq x)$',fontsize=12)
ax2.set_title('Cumulative \n Distribution Function',fontsize=14)

fig.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.9,
                    wspace=0.25)