
from scipy.optimize import minimize
from math import *
import numpy as np
import pandas as pd
import tqdm

class SPOT:
    def __init__(self, q = 1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0
    
    def fit(self,init_data,data):
        if isinstance(data,list):
            self.data = np.array(data)
        elif isinstance(data,np.ndarray):
            self.data = data
        elif isinstance(data,pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return
            
        if isinstance(init_data,list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data,np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data,pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data,int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data,float) & (init_data<1) & (init_data>0):
            r = int(init_data*data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def initialize(self, level = 0.98, verbose = False):
        level = level-floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold 
        self.Nt = self.peaks.size
        self.n = n_init
        g,s,l = self._grimshaw()
        self.extreme_quantile = self._quantile(g,s)
        return 

    def _rootsFinder(fun,jac,bounds,npoints,method):
        if method == 'regular':
            step = (bounds[1]-bounds[0])/(npoints+1)
            X0 = np.arange(bounds[0]+step,bounds[1],step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0],bounds[1],npoints)
        
        def objFun(X,f,jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g+fx**2
                j[i] = 2*fx*jac(x)
                i = i+1
            return g,j
        
        opt = minimize(lambda X:objFun(X,fun,jac), X0, method='L-BFGS-B', jac=True, bounds=[bounds]*len(X0))
        X = opt.x
        np.round(X,decimals = 5)
        return np.unique(X)
    
    def _log_likelihood(Y,gamma,sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma/sigma
            L = -n * log(sigma) - ( 1 + (1/gamma) ) * ( np.log(1+tau*Y) ).sum()
        else:
            L = n * ( 1 + log(Y.mean()) )
        return L

    def _grimshaw(self,epsilon = 1e-8, n_points = 10):
        def u(s):
            return 1 + np.log(s).mean()
            
        def v(s):
            return np.mean(1/s)
        
        def w(Y,t):
            s = 1+t*Y
            us = u(s)
            vs = v(s)
            return us*vs-1
        
        def jac_w(Y,t):
            s = 1+t*Y
            us = u(s)
            vs = v(s)
            jac_us = (1/t)*(1-vs)
            jac_vs = (1/t)*(-vs+np.mean(1/s**2))
            return us*jac_vs+vs*jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()
        a = -1/YM
        if abs(a)<2*epsilon:
            epsilon = abs(a)/n_points
        
        a = a + epsilon
        b = 2*(Ymean-Ym)/(Ymean*Ym)
        c = 2*(Ymean-Ym)/(Ym**2)
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks,t), lambda t: jac_w(self.peaks,t), (a+epsilon,-epsilon), n_points,'regular')
        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks,t), lambda t: jac_w(self.peaks,t), (b,c), n_points,'regular')
        zeros = np.concatenate((left_zeros,right_zeros))
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks,gamma_best,sigma_best)
        for z in zeros:
            gamma = u(1+z*self.peaks)-1
            sigma = gamma/z
            ll = SPOT._log_likelihood(self.peaks,gamma,sigma)
            if ll>ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll
        return gamma_best,sigma_best,ll_best

    def _quantile(self,gamma,sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma/gamma)*(pow(r,-gamma)-1)
        else:
            return self.init_threshold - sigma*log(r)

    def run_simp(self, with_alarm = True):
        th = []
        suoyin = np.array(range(self.data.shape[0]))
        dayu = suoyin[self.data > self.init_threshold]
        self.n = self.data.size
        self.peaks = np.append([], self.data[self.data > self.init_threshold] - self.init_threshold)
        self.Nt = self.data[self.data > self.init_threshold].shape[0]
        g,s,l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)
        th.append(self.extreme_quantile)
        return {'thresholds' : th, 'alarms': []}
