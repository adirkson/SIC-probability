# -*- coding: utf-8 -*-

from scipy.optimize import curve_fit
from scipy.stats import  beta, linregress
import numpy as np
from beinf import beinf

class taqm():
    '''
    Contains the methods needed for performing trend-adjusted quantile mapping (TAQM).
    It relies on the methods from the :class:`beinf` class.
    
    Methods Summary:
    ----------------

    ``calibrate(x_params,y_params,x_t_params,X,Y,X_t,trust_sharp_fcst=False)``  
        calibrated forecast BEINF parameters and calibrated forecast ensemble
        
    ``fit_data(X,Y,X_t)`` 
        BEINF parameters for the TAMH, TAOH, and the raw forecast
        
    ``lin(m,b,T)`` 
        linear equation values
    
    ``piecewise_lin(m1,b1,m2,b2,t_b,T)`` 
        piece-wise linear equation values

    ``trend_adjust_1p(data_all,tau_t,t)``  
        trend-adjusted values using a single period
    
    ``trend_adjust_2p(data_all,tau_t,t,t_b=1999)``  
        trend-adjusted values using two periods
    
    '''

    def lin(self,a1,b1,T):
        r"""Evaluates the piece-wise linear equation
        
            .. math::       
               z = a_1 T + b_1                           
               :label: pw1
               
        at :math:`T`.
        
        Args:
            a1 (float):
                The slope in :eq:`pw1`.
            
            b1 (float):
                The z-intercept in
                :eq:`pw1`.
            
            t_b (float):
                The breakpoint for :math:`z` in :eq:`pw1`.
            
            T (float or ndarray):
                The point(s) at which :eq:`pw1` is evaluated.

        Returns: z (float or ndarray):
            The value of :eq:`pw1` at each point T. Values 
            less than 0 and greater than 1
            are clipped to 0 and 1, respectively.
        """

        #Linear equation
        z = a1*T + b1               
        #don't allow for sic values less than zero or greater than 1
        if np.any(z<0.0):
            if isinstance(z,np.float64):
                z = 0.0
            else:
                z[z<0.0] = 0.0
                
        if np.any(z>1.0):
            if isinstance(z,np.float64):
                z = 1.0 
            else:
                z[z>1.0] = 1.0
                
        return z

    def piecewise_lin(self,a1,b1,a2,b2,t_b,T):
        r"""Evaluates the piece-wise linear equation
        
            .. math::       
               z = \begin{cases} 
                       a_1 T + b_1, & T<t_b \\
                       a_2 T + b_2, & T>t_b
                   \end{cases}                           
               :label: pw2
               
        at :math:`T`.
        
        Args:
            a1, a2 (floats):
                The slopes in :eq:`pw2`.
            
            b1, b2 (floats):
                The z-intercepts in
                :eq:`pw2`.
            
            t_b (float):
                The breakpoint for :math:`z` in :eq:`pw2`.
            
            T (float or ndarray):
                The point(s) at which :eq:`pw2` is evaluated.

        Returns: z (float or ndarray):
            The value of :eq:`pw2` at each point T. Values 
            less than 0 and greater than 1
            are clipped to 0 and 1, respectively.
        """
        if isinstance(T,np.ndarray):
            #If T is an, z and T are arrays
            z = np.zeros(T.shape)
            z[T<t_b] = a1*T[T<t_b] + b1
            z[T>=t_b] = a2*T[T>=t_b] + b2
        else:
            #If for a single year T and y are single value integers or floats
            if T<t_b:
                z = a1*T + b1
            else:
                z = a2*T + b2
                
        #don't allow for sic values less than zero or greater than 1
        if np.any(z<0.0):
            if isinstance(z,np.float64):
                z = 0.0
            else:
                z[z<0.0] = 0.0
                
        if np.any(z>1.0):
            if isinstance(z,np.float64):
                z = 1.0 
            else:
                z[z>1.0] = 1.0
                
        return z 

    def trend_adjust_1p(self,data_all,tau_t,t):
        """Linearly detrend data_all and re-center about its
        linear least squares fit evaluated at 
        :math:`T=t`. One may want to use
        this trend adjustment over :func:`~taqm.trend_adjust_2p` 
        if the hindcast record is over the more recent record.
        
        Args:
            data_all (ndarray):
                A time series of size N, or an ensemble time series of size NxM,
                where M is the number of ensemble members.
            
            tau_t (ndarray):
                All hindcast years exluding the forecast year.

            t (float):
                The forecast year.
                
        Returns: data_ta (ndarray):
            Trend-adjusted values with same shape as data_all.
        """
        if np.ndim(data_all)>1:
            # If data_all is shape N by M, take mean across axis M observations (ensemble mean) 
            # for subsequent calculation of the linear least squares solution
            data = np.mean(data_all,axis=1)
        else:
            data = np.copy(data_all)    
            
        
        if np.all(data_all==0.0) or np.all(data_all==1.0):
            # If all values in data_all are either 0 or 1,
            # no trend adjustment is needed
            data_ta = np.copy(data_all)

        else:
                
            #compute least squares paramaters
            m, b  = linregress(tau_t,data)[:2]
            
            #non-stationary mean for year t                 
            z_tilde_nsm = self.lin(m,b,t)
                
            if z_tilde_nsm<0.0:
                z_tilde_nsm = 0.0
            if z_tilde_nsm>1.0:
                z_tilde_nsm = 1.0
            
            #detrend data and re-center
            if np.ndim(data_all)>1:
                data_d = data_all - self.lin(m,b,tau_t)[:,np.newaxis] 
                data_ta = data_d + z_tilde_nsm
            else:
                data_d = data_all - self.lin(m,b,tau_t)
                data_ta = data_d + z_tilde_nsm
                
            #change values below zero (above one) to zero (one)
            data_ta[data_ta<0.0] = 0.0
            data_ta[data_ta>1.0] = 1.0 
            
        return data_ta

    def trend_adjust_2p(self,data_all,tau_t,t,t_b=1999):
        """Linearly detrend data_all and re-center it about its
        non-linear least squares fit to Eq. :eq:`pw2` evaluated at 
        :math:`T=` `t`. This method carries
        out the trend-adjustment technique described in section 5a
        of Dirkson et al, 2018
        
        Args:
            data_all (ndarray):
                A time series of size N, or an ensemble time series of size NxM,
                where M is the number of ensemble members.
            
            tau_t (ndarray):
                All hindcast years exluding the forecast year `t`.

            t (float):
                The forecast year.
                
            t_b (float):
                The breakpoint year in Eq. :eq:`pw2`.
                
        Returns: data_ta (ndarray):
            Trend-adjusted values with same shape as data_all.
        """
        if np.ndim(data_all)>1:
            # If data_all is shape N by M, take mean across axis M observations (ensemble mean) 
            # for subsequent calculation of the linear least squares solution
            data = np.mean(data_all,axis=1)
        else:
            data = np.copy(data_all)    
            
        
        if np.all(data_all==0.0) or np.all(data_all==1.0):
            # If all values in data_all are either 0 or 1,
            # no trend adjustment is needed
            data_ta = np.copy(data_all)

        else:
            # else, compute non-linear least squares 
            # parameters of the piece-wise equation
            # and recenter data_all about this solution evaluated 
            # at time=year 
            def piecewise_regress(T,z):
                # function for computing least squares parameters
                def f_min(T,a1,b1,a2):
                    # function for the piecewise linear equation
                    # with continuity at yr_bp
                    z_tilde = np.zeros(len(T))
                    z_tilde[T<=t_b] = a1*T[T<=t_b] + b1
                    z_tilde[T>t_b] = a2*T[T>t_b] + (a1-a2)*t_b + b1
                    return z_tilde
    
                popt, pcov = curve_fit(f_min,T,z,p0=[1,1,1])
                a1,b1,a2 = popt
                b2 = (a1-a2)*t_b + b1
                
                return a1,b1,a2,b2
                
            #compute least squares paramaters
            a1, b1, a2, b2  = piecewise_regress(tau_t,data)
            
            #non-stationary mean for year t                 
            z_tilde_nsm = self.piecewise_lin(a1,b1,a2,b2,t_b,t)
                
            if z_tilde_nsm<0.0:
                z_tilde_nsm = 0.0
            if z_tilde_nsm>1.0:
                z_tilde_nsm = 1.0
            
            #detrend data and re-center
            if np.ndim(data_all)>1:
                data_d = data_all - self.piecewise_lin(a1,b1,a2,b2,t_b,tau_t)[:,np.newaxis] 
                data_ta = data_d + z_tilde_nsm
            else:
                data_d = data_all - self.piecewise_lin(a1,b1,a2,b2,t_b,tau_t)
                data_ta = data_d + z_tilde_nsm
                
            #change values below zero (above one) to zero (one)
            data_ta[data_ta<0.0] = 0.0
            data_ta[data_ta>1.0] = 1.0 
            
        return data_ta
       
    def fit_params(self,X,Y,X_t):
        '''
        Fits X (the TAMH ensemble time series), Y (the TAOH time series), and X_t (the raw forecast ensemble) to the BEINF
        distribution. This method carries out the fiting procedure described in 
        section 5b of Dirkson et al, 2018.
        
        Args:
            X (ndarray):
                The TAMH ensemble time series of size NxM.
                
            Y (ndarray):
                The TAOH time series of size N.
                
            X_t (ndarray):
                The raw forecast ensemble of size M.
                
        Returns: x_params (ndarray), y_params (ndarray), x_t_params (ndarray):
            The shape parameters :math:`a`,
            :math:`b`, :math:`p`, :math:`q`
            for the BEINF distribution fitted to each  
            X, Y, and X_t  (see :meth:`beinf.fit`).             
        '''
        #Fit TAMH to the beinf distribution
        a_x, b_x, p_x, q_x = beinf.fit(X.flatten()) 
        x_params = np.array([a_x, b_x, p_x, q_x])

        #fit TAOH to the beinf distribution
        a_y, b_y, p_y, q_y = beinf.fit(Y)     
        y_params = np.array([a_y, b_y, p_y, q_y])

        #fit forecast to the beinf distribution
        a_x_t, b_x_t, p_x_t, q_x_t = beinf.fit(X_t)
        #store parameters in an array
        x_t_params = np.array([a_x_t, b_x_t, p_x_t, q_x_t])
        
        return x_params, y_params, x_t_params


    def calibrate(self,x_params,y_params,x_t_params,X,Y,X_t,trust_sharp_fcst=False):
        r'''
        Calibrates the raw forecast BEINF paramaters :math:`a_{x_t}`,
        :math:`b_{x_t}`, :math:`p_{x_t}` and :math:`q_{x_t}`. This method 
        carries out the calibration step described in section 5c in Dirkson et al, 2018.
        
        Args:
            x_params (ndarray):
                An array containing the four parameters of the BEINF distribution
                for the TAMH ensemble time series.

            y_params (ndarray):
                An array containing the four parameters of the BEINF distribution
                for the TAOH time series.
                
            x_t_params (ndarray):
                An array containing the four parameters of the BEINF distribution
                for the raw forecast ensemble.
             
            trust_sharp_fcst (boolean, optional):
                `True` to revert to the raw forecast when 
                :math:`p_{x_t}=1`. `False` 
                to revert to the TAOH distribution when :math:`p_{x_t}=1`.
                
        Returns: x_t_cal_params (ndarray), X_t_cal_beta (ndarray):
            x_t_cal_params contains the four BEINF
            distribution parameters for the calibrated forecast: :math:`a_{\hat{x}_t}`, :math:`b_{\hat{x}_t}`, 
            :math:`p_{\hat{x}_t}` and :math:`q_{\hat{x}_t}`. When  :math:`a_{\hat{x}_t}` and :math:`b_{\hat{x}_t}`
            could not be fit, they are returned as :code:`a=np.inf` and :code:`b=np.inf`.
            
            X_t_cal_beta contains the calibrated forecast ensemble (np.inf replaces replace 0's and 1's in the 
            ensemble). This array contains all :code:`np.inf` values when any of :math:`p_y=1`,
            :math:`p_x=1`, or :math:`p_{x_t}=1`, or when all parameters in x_t_cal_params are defined (none are equal to :code:`np.inf`).
            
        '''
        
        a_x, b_x, p_x, q_x = x_params[0], x_params[1], x_params[2], x_params[3]
        a_y, b_y, p_y, q_y = y_params[0], y_params[1], y_params[2], y_params[3]
        a_x_t, b_x_t, p_x_t, q_x_t = x_t_params[0], x_t_params[1], x_t_params[2], x_t_params[3]
        
        #function to avoid returning nan when dividing by zero
        def safediv(numerator,denominator):
            if denominator==0.0:
                return 0.0
            else:
                return numerator/denominator             
                       
        #calibrate forecast parameters
        if p_y==1.0 or p_x==1.0 or p_x_t==1.0:
            # if any of TAMH, TAOH, or the forecast are entirely
            # comprised of zeros and ones, calibration cannot be done.
            # choices are: 
            if np.logical_and(p_x_t==1.0,trust_sharp_fcst==True):
                # trust the raw forecast values when they are perfectly sharp
                a_x_t_cal, b_x_t_cal, p_x_t_cal, q_x_t_cal = np.copy(a_x_t), np.copy(b_x_t), np.copy(p_x_t), np.copy(q_x_t)      
            else:
                # trust TAOH
                a_x_t_cal, b_x_t_cal, p_x_t_cal, q_x_t_cal = np.copy(a_y), np.copy(b_y), np.copy(p_y), np.copy(q_y)
            #in this case, there are no "beta" claibrated forecast values
            X_t_cal_beta = np.inf*np.ones(len(X_t))
        else:
            #calibrate the bernoulli portion of the beinf forecast distribution
            p_x_t_cal = max(min(p_x_t + p_y - p_x,1.),0.)                    
            num = max(min(p_x_t*q_x_t + p_y*q_y - p_x*q_x,1.),0.)
            den = np.copy(p_x_t_cal)
            q_x_t_cal = max(min(safediv(num,den),1.),0.) 

            if p_x_t_cal==1.0:
                # if the calibration of p and q result in p=1 then there
                # should be no "beta" calibrated forecast values
                a_x_t_cal,b_x_t_cal = np.inf, np.inf
                X_t_cal_beta = np.inf*np.ones(len(X_t))
            else:            
                # Get the values from the forecast ensmeble TAMH, and 
                # TAOH that are between zero and one
                X_t_beta = np.copy(X_t[np.logical_and(X_t!=0.0,X_t!=1.0)]) 
                X_beta = X[np.logical_and(X!=0.0,X!=1.0)].flatten()
                Y_beta = Y[np.logical_and(Y!=0.0,Y!=1.0)]                           

                # calibrate forecast SIC values 
                # between zero and one using quantile mapping
                if a_x!=np.inf and a_y!=np.inf and a_x_t!=np.inf:
                    rv_x_beta = beta(a_x, b_x, loc=0.0, scale=1.0 - 0.0)
                    rv_y_beta = beta(a_y, b_y, loc=0.0, scale=1.0 - 0.0)
                    # if none of cases 2-4 were encountered
                    # for the raw forecast values
                    X_t_cal_beta = rv_y_beta.ppf(rv_x_beta.cdf(X_t_beta)) 
                else:
                    #revert to empirical-based quantile mapping
                    X_t_cal_beta = np.percentile(Y_beta,beinf.ecdf(X_t_beta,X_beta)*100.,interpolation='linear')
            
                cases = beinf.check_cases(X_t_cal_beta)
                #Solving of Eq. 8 (quantile mapping)
                if cases==False:                
                    a_x_t_cal, b_x_t_cal = beinf.fit_beta(X_t_cal_beta) 
                    X_t_cal_beta = np.inf*np.ones(len(X_t))
                else:
                    a_x_t_cal, b_x_t_cal = np.inf, np.inf
                           
        if len(X_t)!=len(X_t_cal_beta):
            X_t_cal_beta = np.append(X_t_cal_beta,np.inf*np.ones(len(X_t)-len(X_t_cal_beta)))

        x_t_cal_params = np.array([a_x_t_cal, b_x_t_cal, p_x_t_cal, q_x_t_cal])
        #Return BEINF distribution parameters  
        return x_t_cal_params, X_t_cal_beta

    def unpack_params(self,array):
        a,b,p,q = map(lambda i: array[i], range(len(array)))
        return a,b,p,q