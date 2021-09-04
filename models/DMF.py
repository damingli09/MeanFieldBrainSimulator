#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Single population dynamic mean-field (DMF) model for large scale simulation"""
__author__ = "Daming Li"
__email__ = "daming.li@yale.edu"

import numpy as np
import math
import time
from scipy.stats import pearsonr
from scipy.signal import fftconvolve
from sklearn.preprocessing import normalize
from statsmodels.tsa.api import acf
import scipy.optimize as opt
from obata04 import *
from sklearn.decomposition import PCA
from DMF_params import *
from BaseModel import BaseModel

class DMF(BaseModel):
    """
    Implements the DMF model defined in Deco JNeurosci (2014) and Demirtas Neuron (2018) papers.
    """
    def __init__(self, SC, FC=None, wEE=1.5,  SgE=0., scaling=1.0, gE=1.0, threshold=108., sigma=0.05, I_ext = 0.3, row_normalize=True):
        super(DMF, self).__init__(SC,FC,sigma,row_nomalize)
        if isinstance(I_ext, (int,float)):
            self.I_ext = I_ext*np.ones(n)
        else:
            self.I_ext = I_ext
        self.wEE = wEE
        self.SgE = SgE
        self.scaling = scaling # multiplicative gain modulation
        self.gE = gE*gainconst
        self.threshold = threshold

    def FP_x(self):
        '''
        Computes the (three) fixed points in a list and returns the x coordinate of the low and high state
        '''
        
        # define the right hand of wilson-cowan equations
        def my_WCr(x):
            
            S = x[0]
            a = x[1]
            
            R = F(I=self.wEE*JN*S-a+self.I_ext[0], g=self.gE, b=self.threshold)
            yS = -S/tauS + (1-S)*saturation*R
            ya = -a + JA*R
            y = np.array([yS, ya])
            
            return y
    
        res = []
        
        x0 = np.array([0.0, 0.0])
        res.append(opt.root(my_WCr, x0).x)
        x0 = np.array([0.5, 0.0])
        res.append(opt.root(my_WCr, x0).x)
        x0 = np.array([1.0, 0.0])
        res.append(opt.root(my_WCr, x0).x)
        
        x0 = np.array([0.0, 0.5])
        res.append(opt.root(my_WCr, x0).x)
        x0 = np.array([0.5, 0.5])
        res.append(opt.root(my_WCr, x0).x)
        x0 = np.array([1.0, 0.5])
        res.append(opt.root(my_WCr, x0).x)

        res.sort(key = lambda x: x[0])
        
        return res[0][0], res[len(res)-1][0]
    
    def nullclines(self, I_tot=np.arange(0.2, 0.6, 0.001)):
        '''
        Computes the S and a nullclines of the system
        '''

        x = I_tot[I_tot<=(self.wEE*JN + self.I_ext[0])]
        R = F(x, g=self.gE, b=self.threshold)

        Ss = saturation*R/(1./tauS + saturation*R)  # s coordinate of S-nullcline
        Sa = self.wEE*JN*Ss + self.I_ext[0] - x  # a coordinate of S-nullcline

        Aa = JA*R  # s coordinate of a-nullcline
        As = (x+Aa-self.I_ext[0])/(self.wEE*JN)  # a coordinate of a-nullcline

        return Ss, Sa, As[(Aa>=0)&(As<=Ss.max())], Aa[(Aa>=0)&(As<=Ss.max())]

    def integrate(self, T=300, randseed=0, signal=False, verbose=False):
        """
        Runs the simulation.

        Updates
        -------
            all the dynamic and BOLD variables
        """

        n = len(self.SC)
        Lt = int(T/dt)
        np.random.seed(int(time.time())+randseed*17)
        
        XE = np.zeros((n,Lt),dtype=float)
        Mu = mu * np.ones(n,dtype=float)
        for i in range(Lt-1):
            XE[:,i+1] = XE[:,i] + dt/tauOU * (Mu-XE[:,i]) + np.sqrt(2.*dt/tauOU)*np.random.randn(n)*self.sigma
        
        XE = np.random.randn(n,Lt)*self.sigma  # Gaussian noises
        
        # if signal is needed, add to XE on somatosensory areas (range(8,15)), from T=40 to T= 100
        strength = 1.0  # signal strength, assuming constant
        if signal:
            #for i in range((40*2000),(50*2000)):  # T=40 to T=50
                #XE[8:15,i] += strength*(i-80000)/20000  # somatosensory areas
            for i in range((20*resolution),int(20.01*resolution)):  # T=20 to T=21
                XE[8:15,i] += strength  # somatosensory areas

        # initialize firing rate
        self.S = np.zeros((n,Lt),dtype=float)
        self.R = np.zeros((n,Lt),dtype=float)
        self.a = np.zeros((n,Lt),dtype=float)  # adaptation
        self.S[:,0] = 0.0*np.ones(n)
        self.R[:,0] = 0.0*np.ones(n)
        self.a[:,0] = 0.0*np.ones(n)
        ## Biophysical BOLD
        x = np.zeros((n,Lt),dtype=float)
        f = np.zeros((n,Lt),dtype=float)
        v = np.zeros((n,Lt),dtype=float)
        q = np.zeros((n,Lt),dtype=float)
        self.y = np.zeros((n,Lt),dtype=float)
        f[:,0] = np.ones(n)
        v[:,0] = np.ones(n)
        q[:,0] = np.ones(n)
        
        for i in range(1,Lt):
            ## dynamics
            self.S[:,i] = self.S[:,i-1] + dt * (-self.S[:,i-1]/tauS + (1 - self.S[:,i-1])*saturation*(self.R[:,i-1]) + XE[:,i])
            self.a[:,i] = self.a[:,i-1] + dt/tauA * (-self.a[:,i-1] + JA*self.R[:,i-1])
            #self.R[:,i] = self.scaling*F(I=self.wEE*JN*(self.S[:,i])+ JN*self.SgE*(self.SC).dot(self.S[:,i]) - self.a[:,i] + self.I_ext + XE[:,i], g=self.gE, b=self.threshold)
            self.R[:,i] = self.scaling*F(I=self.wEE*JN*(self.S[:,i])+ JN*self.SgE*(self.SC).dot(self.S[:,i]) - self.a[:,i] + self.I_ext, g=self.gE, b=self.threshold)
            ## Biophysical BOLD
            x[:,i] = x[:,i-1] + dt*(self.S[:,i-1] - kappa*x[:,i-1] - gamma*(f[:,i-1] - 1))
            f[:,i] = f[:,i-1] + dt*x[:,i-1]
            v[:,i-1] = v[:,i-1].clip(min=0.0001)
            v[:,i] = v[:,i-1] + dt/tau * (f[:,i-1] - v[:,i-1]**(1.0/alpha))
            q[:,i] = q[:,i-1] + dt/tau * (f[:,i-1]/rho * (1 - (1-rho)**(1.0/f[:,i-1])) - q[:,i-1]*(v[:,i-1]**(1.0/alpha-1)))

            self.y[:,i] = V0*(k1*(1-q[:,i]) + k2*(1-q[:,i]/v[:,i]) + k3*(1-v[:,i]))

    


    










