# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rv_continuous, beta, binom
from scipy import special

class beinf_gen(rv_continuous):            
    r"""A zero- and one- inflated beta (BEINF) random variable
       
    Notes
    -----
    The probability density function for BEINF is:
        .. math::
        
           f(x;a,b,p,q) = p(1-q)\delta(x)+(1-p)f_{\mathrm{beta}}(x;a,b) + pq\delta(x-1)
                                   
    where
    
        .. math::
            
           f_{\mathrm{beta}}(x;a,b)=\frac{1}{\mathrm{B}(a,b)}x^{a-1}(1-x)^{b-1},
     
    :math:`\delta(x)` is the delta function, and :math:`\mathrm{B}(a,b)` is the 
    beta function (:py:data:`scipy.special.beta`).
    
    
    :class:`beinf` takes :math:`a>0`, :math:`b>0`, :math:`0\leq p \leq 1`, :math:`0\leq q \leq 1`
    as shape parameters.    


    :class:`beinf` is an instance of a subclass of :py:class:`~scipy.stats.rv_continuous`, and therefore
    inherits all of the methods within :py:class:`~scipy.stats.rv_continuous`. Some of those methods
    have been subclassed here:
    
    ``_argcheck``

    ``_cdf``    
    
    ``_pdf``

    ``_ppf``

    ``_stats``

    
    Additional methods added to :class:`beinf` are:

    ``beta_moments(data_sub)``
        Computes the parameters for the beta distribution using the method
        of moments.        
        
    ``cdf_eval(x,data_params,data)``
        Chooses the appropriate method for evaluating the cumulative
        distribution function for data at points x, and computes it.

    ``check_cases(data)``
        Checks to see if data satisfies any of cases 1-4.
        
    ``ecdf(x, X_samp, p=None, q=None)``
        The empirical distribution function X_samp at points x.
    
    ``fit(data)``
        This replaces the ``fit`` method in :py:class:`~scipy.stats.rv_continuous`, but
        is not subclassed. Computes the MLE parameters for the BEINF distribution.
        
    ``fit_beta(data_sub)``
        Computes the MLE parameters for the beta distribution.
    
    """

#####################################################################    
######################### Subclassed Methods ########################
#####################################################################

    def _pdf(self, x, a, b, p, q):
        '''
        Subclass the _pdf method (returns the pdf of the 
        BEINF distribution)
        '''
        def logpdf_beta(x,a,b): 
            # log of the pdf for beta distribution (taken from scipy.stats.beta)
            lPx = special.xlog1py(b-1.0, - x) + special.xlogy(a-1.0, x)
            lPx -= special.betaln(a, b)
            return lPx
        
        def delta(x):
            if isinstance(x,np.float):
                #if x comes in as float, turn it into a numpy array
                x = np.array([x])
            
            y = np.zeros(x.shape)
            y[x==0.0] = 1.0
            return y
           
        vals = delta(x)*p*(1-q) + (1.-p)*np.exp(logpdf_beta(x,a,b)) - p*q*delta(x-1.)
        # boolean list for random variable
        return vals
        
    def _cdf(self, x, a, b, p, q):       
        if np.all(p==1.):
            #cdf is only described by bernoulli distribution when p=1
            return p*binom._cdf(x,1,q)
        else:    
            # the ranges of x that break up the piecewise
            # cdf
            condlist = [x<0.0,
                        x==0.0, 
                        np.logical_and(x>0.0, x<1.0),
                        x>=1.0]
            # the piecewise cdf associated with the entries in condlist
            choicelist = [0.0,
                          p*(1-q),
                          p*binom._cdf(x,1,q) + (1-p)*special.btdtr(a,b,x),
                          1.0]
            
            return np.select(condlist, choicelist)


    def _ppf(self, rho, a, b, p, q):  
        # subclass the _ppf method (returns the inverse cdf of the
        # beinf distribution). NOTE: This is not a true inverse
        # when p!=0 since the same SIC value can be returned
        # for a range of probabilities at the endpoints
        if np.all(p==1):
            gamma_ = 0.0
        else:
            gamma_ = (rho - p*(1-q))/(1-p)
            
        condlist=[np.logical_and(rho>=0., rho<=p*(1-q)), 
              np.logical_and(rho>p*(1-q),rho<(1-p*q)),
              np.logical_and(rho>=1-p*q, rho<=1.)]
              
        choicelist=[0.0, special.btdtri(a,b,gamma_), 1.0]
              
        return np.select(condlist, choicelist)

                
    def _stats(self, a, b, p, q, moments='mv'):       
        if np.all(p==1.0):
            # When p=1, there is only a mass point at 0 and/or 1 and
            # the statistical moments are 
            # computed from the bernoulli distribution

            prob_1 = q
            prob_0 = 1-q       
            mn = prob_1 #mean
            var = prob_0*prob_1  #variance

            if np.sqrt(var)==0.0:
                skew = 0.0 #force skewness=0 when variance=0
            else:
                skew = (1-2*prob_1) / np.sqrt(var) #skewness
            
            if var==0.0:
                kurt = 0.0 # force kurtosis=0 when variance=0            
            kurt = (1.0 - 6*var) / var #kurtosis

        else:        
            #bernoulli statistical moments
            prob_1 = q
            prob_0 = 1-q       
            mu_bern = prob_1 #mean
            var_bern = prob_0*prob_1  #variance

            if np.sqrt(var_bern)==0.0:
                skew_bern = 0.0 #force skewness=0 when variance=0
            else:
                skew_bern = (1-2*prob_1) / np.sqrt(var_bern) #skewness
            
            if var_bern==0.0:
                kurt_bern = 0.0 # force kurtosis=0 when variance=0  
            else:
                kurt_bern = (1.0 - 6*var_bern) / var_bern #kurtosis
    
            #beta statistical moments
            mu_beta = a*1.0 / (a + b)   #mean
            var_beta = a*b/((a+b)**2.*(a+b+1.))
            skew_beta = 2.0*(b-a)*np.sqrt((1.0+a+b)/(a*b)) / (2+a+b)    #skewness
       
            #kurtosis
            kurt_beta = 6.0*(a**3 + a**2*(1-2*b) + b**2*(1+b) - 2*a*b*(2+b))
            kurt_beta /= a*b*(a+b+2)*(a+b+3)        
    
            # In the case that p!=1.0, the statistics are computed 
            # as weighted averages of the statistics for the
            # bernoulli and beta distributions, respectively by p and 1-p
           
            # Mean of beinf 
            mn = p*mu_bern + (1 - p)*mu_beta

            # Variance of beinf
            var = p*var_bern + (1 - p)*var_beta
            
            # skewness (weighted sum of skew for beta and bernoulli)
            skew = p*skew_bern + (1 - p)*skew_beta
            # kurtosis (wighted sum of kurtosis for beta and bernoullie)
            kurt = p*kurt_bern + (1 - p)*kurt_beta
      
        return mn, var, skew, kurt

    def _argcheck(self,a,b,p,q):
        #subclass the argcheck method to ensure parameters
        #are constrained to their bounds
        check1 = np.logical_and(a>=0.,b>=0.)
        check2 = np.logical_and(p>=0.,p<=1.)
        check3 = np.logical_and(q>=0.,q<=1.)

        if check1==True and check2==True and check3==True:
            return True
        else:
            return False

#####################################################################    
######################### Methods Unique ############################
#########################  to the beinf  ############################
#########################  distribution  ############################
#####################################################################

    def cdf_eval(self,x,data_params,data):
        r'''
        Evaluates the cumulative distribution function (empirically or parametrically)
        for a random variable :math:`X\sim\mathrm{BEINF}(a,b,p,q)`. When
        :math:`a` and :math:`b` are unkown (i.e. when :code:`a=np.inf`
        and :code:`b=np.inf`), 
        the cdf is evaluated using :meth:`~beinf.beinf_gen.ecdf`. When :math:`a` 
        and :math:`b` are known `or` when :math:`p=1`, the cdf is 
        evaluated using :py:meth:`~scipy.stats.rv_continuous.cdf`. This is used
        after applying TAQM (see the tutorial on using the :class:`taqm` class).
        
        Args:
            x (float or ndarray):
                The value(s) at which the ecdf is evaluated.
                
            data_params (ndarray):
                An array containing the four parameters a, b, p, and q.
                
            data (ndarray):
                The data which defines the ecdf when :math:`a` and :math:`b`
                are unknown (i.e. when :code:`a=np.inf` and :code:`b=np.inf`).
                This array may contain the values :code:`np.inf`,
                0, and 1.
                
        Returns: cdf_vals (ndarry):       
            The cdf or ecdf for :math:`X\sim\mathrm{BEINF}(a,b,p,q)`
            evaluated at x.
        '''
        a,b,p,q = data_params[0],data_params[1],data_params[2],data_params[3]
        if a==np.inf and p!=1.0:
            cdf_vals = beinf.ecdf(x,data,p,q)
        else:
            rv = beinf(a,b,p,q)
            cdf_vals = rv.cdf(x)
            
        return cdf_vals

    def ecdf(self, x, X_samp, p=None, q=None):
        r'''
        For computing the empirical cumulative distribution function (ecdf) 
        for a random variable :math:`X\sim\mathrm{BEINF}(a,b,p,q)` when the parameters 
        :math:`a` and :math:`b` (or additionally :math:`p` and :math:`q`) are unkown.
        
        Args:
            x (float or ndarray):
                The value(s) at which the ecdf is evaluated
               
            X_samp (float or ndarray):
                A sample that lies on the either: the open
                interval (0,1) when p and q **are** included as
                arguments, `or` the closed interval
                [0,1] when p and q **are not** included as arguments. 
                
            p (float, optional):
                Shape parameter(s) for the beinf distribution. 
                
            q (float, optional):
                Shape parameter(s) for the beinf distribution. 
                
        Returns: ecdf_vals (ndarray):            
            The ecdf for X_samp, evaluated at x.
            
        '''
        
        if isinstance(x,np.float):
            #if x comes in as float, turn it into a numpy array
            x = np.array([x])

        if isinstance(X_samp,np.float):
            #if X_beta comes in as float, turn it into a numpy array
            X_samp = np.array([X_samp])  
    
       
        if p!=None and q!=None:
            # sort the values of X_beta from smallest to largest
            # but take out 0's and 1's because p and q take care of endpoints
            xs = np.sort(X_samp[np.logical_and(X_samp!=0.0,X_samp!=1.0)])
            # also take out any inf values 
            xs = np.sort(xs[xs!=np.inf])
            
            # get the sample size of xs satisfying xs<=x for each x
            ys = map(lambda vals: len(xs[xs<=vals]), x)

            # the ranges of x that break up the piecewise
            # cdf
            condlist = [x<0.0,
                    x==0.0, 
                    np.logical_and(x>0.0, x<1.0),
                    x>=1.0] 
            # the piecewise cdf associated with the entries in condlist
            choicelist = [0.0,
                          p*(1-q),
                          p*(1-q) + (1-p)*np.array(ys)/float(len(xs)),
                          1.0]
        else:
            # sort the values of X_beta from smallest to largest
            xs = np.sort(X_samp)
            
            # get the sample size of xs satisfying xs<=x for each x
            ys = map(lambda vals: len(xs[xs<=vals]), x)
            
            # the ranges of x that break up the piecewise
            # cdf
            condlist = [x<0.0,
                        np.logical_and(x>=0.0,x<=1.0)] 
            # the piecewise cdf associated with the entries in condlist
            choicelist = [0.0,
                          np.array(ys)/float(len(xs))]               
        
        ecdf_vals = np.select(condlist, choicelist)
        return ecdf_vals


        
        
    def fit(self, data):      
        '''
        Computes the MLE parameters :math:`a`, :math:`b`, :math:`p`, 
        and :math:`q` for the BEINF distribution to the given data. 
        
        Args:
            data (ndarray):
                The data to be fit to the BEINF distribution.
                
        Returns: a,b, p, q (floats):
            The shape parameters for the BEINF distribution. When :math:`a`
            and :math:`b` cannot be fit, they are returned as 
            :code:`a=np.inf` and :code:`b=np.inf`.
        '''        
        
        data = np.copy(data)
        n=len(data)

        # Compute mle estimates p and q based on the complete sample

        #indicator function
        I = np.zeros(n)
        I[np.logical_or(data==0.0,data==1.0)] = 1.0
        
        # p is the fraction of the sample that contains either
        # 0 or 1
        p=I.sum()/float(n)
        
        # Of those number of 0's and 1's in the sample, q
        # is the fraction of 1's.
        if p==0.0:
            q = 0.0
        else:
            q=(data*I).sum()/I.sum()


        # Compute estimates for a and b from the sub-sample of data
        # neither zero nor one

        # sub-set of data that lies on (0,1)
        data_sub = data[np.logical_and(data!=0.0,data!=1.0)]

        # see if any of the special cases are true for data 
        cases = self.check_cases(data)        
        if cases==False:
            # when all of the cases are false for data,
            # data_sub can be fit using beinf.fit_beta
            a, b = self.fit_beta(data_sub)        
        else:
            # if any of cases 1-4 are True, don't fit beta portion of
            # the beinf distribution and set a and b equal to infinity
            # (note that these values of a and b can still be passed 
            # into beinf... the resultant random variable  will just
            # be described by the bernoulli distribution
            a,b = np.inf, np.inf
            
        return a,b,p,q
      
    def fit_beta(self,data_sub):
        '''
        Computes the MLE parameters :math:`a` and :math:`b` for the 
        beta distribution to the given data.
        
        Args:
            data_sub (ndarray):
                The data to be fit to the beta distribution. The values in
                data_sub should lie on the open interval (0,1).
        
        Returns: a,b (floats):
            The shape parameters for the beta distribution.
        '''
        # start out with a and b as arbitrary negative values.
        # This is to handle instances when errors are raised using the 
        # beta.fit method. When errors are raised this function sets a
        # and b equal to estimates obtained from the method of moments 

        a=-10
        b=-10

        while a<0.0 or b<0.0:
            try:
                # try to fit the data to the beta distribution
                # using the built-in beta.fit method
                a,b,loc,scale = beta.fit(data_sub,floc=0.0,fscale=1.0-0.0)
                
                if a<0.0 or b<0.0:
                    # this is to deal with when a and b are found
                    # succesfully by beta.fit, but are non-sensical
                    # negative values. In this case use the
                    # method of moments to find a and b
                    a, b = self.beta_moments(data_sub)
                    
            except:
                # this is to deal with when the beta.fit function
                # does not converge. In this case use the method
                # of moments to find a and b
                a, b = self.beta_moments(data_sub)

        return a, b

    def beta_moments(self,data_sub):
        '''
        Computes the method of moments estimates of `a` and `b` for the
        beta distribution.
        
        Args:
            data_sub (ndarray):
                The data to be fit to the beta distribution. The values in 
                data_sub should lie on the open interval (0,1).
        Returns: a,b (floats):
            The shape parameters for the beta distribution.
        '''      
        # geometric mean
        data_bar = data_sub.mean()
        #sample standard devation
        data_var = data_sub.var(ddof=1)
        
        #method-of-moments estimates of a and b
        a = data_bar*(data_bar*(1.-data_bar)/data_var - 1.)
        b = (1-data_bar)*(data_bar*(1.-data_bar)/data_var - 1.)   
        
        return a, b


    def check_cases(self,data):
        r'''
        Checks whether data satisfies any of cases 1-4 described in
        Section 3b of Dirkson et al, 2018. When True, the parameters a 
        and b for the beta-portion of the BEINF distribution cannot be fit.
        
        Args:
            data (ndarray):
                The values for which to test if cases 1-4 apply.
                
        Returns: `True` or `False`
            Boolean value of `True` or `False`. `True` if any of the cases 
            are satisfied; `False` if all of the cases are
            not satisfied.
        '''
        case1,case2,case3,case4 = False, False, False, False
        n = len(data)
        x_sub = data[np.logical_and(data!=0.0,data!=1.0)]
        m = n-len(x_sub) #number of zeros and ones in data
        
        s_bar = x_sub.mean()
        v_bar = np.var(x_sub,ddof=1)
        
        if n-m==0:
            case1 = True
        if n-m==1:
            case2 = True
        if n-m>1 and v_bar<1e-20:
            case3 = True
        if v_bar>=s_bar*(1.-s_bar):
            case4 = True   
            
        if case1==False and case2==False and case3==False and case4==False:
            return False
        else:
            return True

beinf = beinf_gen(a=np.nextafter(0,-1),b=1.0,name='beinf',
                  shapes='a,b,p,q')
