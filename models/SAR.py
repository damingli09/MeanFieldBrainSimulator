#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simultaneous autoregressive model (SAR) for large scale simulation"""
__author__ = "Daming Li"
__email__ = "daming.li@yale.edu"

import numpy as np
import math
from scipy.stats import pearsonr
from scipy.signal import fftconvolve
from sklearn.preprocessing import normalize
from statsmodels.tsa.api import acf
from obata04 import *
from SAR_params import *
from BaseModel import BaseModel

class SAR(BaseModel):
    """
    Implements the rate fluctuation model defined in Messe Neuroimage (2015) paper.
    """
    def __init__(self, SC, FC=None, k=100, sigma=1.0, row_nomalize=True):
        super(SAR, self).__init__(SC,FC,sigma,row_nomalize)
        self.k = k

    def integrate(self, T=500, verbose=False):
        """
        Runs the simulation.

        Updates
        -------
            all the dynamic and BOLD variables
        """

        n = len(self.SC)
        Lt = int(T/dt)

        #nu = self.sigma*np.random.randn(n,Lt)  # Gaussian noises
        XE = self.sigma*np.random.randn(n,Lt)

        # initialize firing rate
        self.rE = np.zeros((n,Lt),dtype=float)
        self.rE[:,0] = np.random.randn(n)

        for i in range(1,Lt):
            ## dynamics
            self.rE[:,i] = self.k*(self.SC).dot(self.rE[:,i-1]) + XE[:,i]




