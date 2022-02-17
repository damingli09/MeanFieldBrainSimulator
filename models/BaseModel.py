#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Builds the base class for large scale simulation"""
__author__ = "Daming Li"
__email__ = "daming.li@yale.edu"


import numpy as np
import math
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize
from tools import *
import time
from scipy.stats import gamma as statgamma
from scipy.signal import fftconvolve
from statsmodels.tsa.api import acf
from sklearn.decomposition import PCA
import scipy.optimize as opt
import itertools
from scipy.signal import hilbert, chirp, butter, filtfilt, welch
from bct.algorithms.modularity import community_louvain
from bct.algorithms.centrality import participation_coef
import statsmodels.api as sm

class BaseModel(object):
    """
    The base class for large scale simulation

    Attributes
    ----------
    SC
        Structural connectivity matrix with rows being 'from' and columns being 'to', and is then transposed and row-normalized
    FC
        Empirical functional connectivity matrix
    sigma
        Noise level
    row_normalize
        Whether to row normalize SC 
    simFC
        Simulated functional connectivity matrix
    bold
        BOLD signals of all ROIs
    """

    def __init__(self, SC, FC, sigma, row_normalize):
        self.SC = SC.T
        n = len(SC)
        for i_area in range(n): self.SC[i_area, i_area] = 0  # zero the diagonals
        if row_normalize: self.SC = normalize(self.SC, axis=1, norm='l1' )  # row-normalize SC
        self.FC = FC
        self.simFC = None
        self.bold = None
        self.sigma = sigma

    def convolveBOLD(self, dynamics, p=2., taub=1., o=0.6):
        """
        Computes BOLD signals by convolving with a given dynamic variable (parcel by time)

        Parameters
        ----------
        p, taub, o
            These are the parameters of the convolution kernel
        """

        self.bold = np.zeros_like(dynamics)
        n = len(self.SC)
        range_t = np.arange(dynamics.shape[1])*self.dt

        fbold_vec = fbold(range_t, p=p, taub=taub, o=o)
        for i in range(n):
            self.bold[i,:] = fftconvolve(dynamics[i,:], fbold_vec, mode='full')[:len(range_t)]

    def setBOLD(self, TS):
        """
        Sets BOLD signals to specific time series such as y or rE

        Parameters
        ----------
        TS
            numpy 2d array of shape (# ROIs, # time steps)
        """

        self.bold = TS

    def butter_bandpass_filter(self, lowcut, highcut, fs, order=2):
        """Performs a low, high or bandpass filter if low & highcut are in range"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        low_in_range = 0 < low < 1
        high_in_range = 0 < high < 1
        if low_in_range and high_in_range:
            b, a = butter(order, [low, high], btype='band')
        elif low_in_range and not high_in_range:
            b, a = butter(order, low, btype='high')
        elif not low_in_range and high_in_range:
            b, a = butter(order, high, btype='low')
        else:
            return #self.bold
        self.bold = filtfilt(b, a, self.bold)

    def GSR(self, global_signal=None):
        """Performs GSR and updates self.bold"""
        if global_signal:
            X = global_signal
        else:
            X = self.bold.mean(axis=0)
        X = sm.add_constant(X)

        n = len(self.SC)
        for i in range(n):
            y = self.bold[i,:]
            reg = sm.OLS(y, X).fit()
            self.bold[i,:] = np.array(reg.resid)

    def computeFC(self):
        """
        Computes simFC
        """

        if self.bold is None: raise Exception("Set BOLD variable first!")

        self.simFC = np.corrcoef(self.bold)

    def fittingScore(self):
        """
        Computes goodness of fit between empirical FC and simulated FC.
        
        Returns
        -------
        double
            Pearsonr between the upper subdiagonals
        """

        if self.simFC is None: self.computeFC()

        return pearsonr(subdiag(self.FC), subdiag(self.simFC))[0]

    def simGBC(self):
        """
        Computes simulated GBC from simFC.

        Returns
        -------
        numpy array
            GBC vector
        """

        if self.simFC is None: raise Exception("simFC is not computed yet!")

        return gbc(self.simFC)

    def fittingScore(self):
        """
        Computes goodness of fit between empirical FC and simulated FC.
        
        Returns
        -------
        double
            Pearsonr between the upper subdiagonals
        """

        if not self.FC: raise Exception("FC is not provided!")
        if self.simFC is None: self.computeFC()

        return pearsonr(subdiag(self.FC), subdiag(self.simFC))[0]

    def varBOLD(self):
        """
        computes variance of BOLD for each ROI

        Returns
        -------
        numpy array of length # ROI
            vector of var of BOLD (for each ROI)
        """

        return np.var(self.bold, axis=1)

    def dimensionality_BOLD(self):
        """
        Computaes the dimensionality of BOLD as ratio of sum of squared singular values over square of summed singular values
        
        Returns
        -------
        double
            Effective dimensionality
        """

        df = np.transpose(self.bold)
        pca = PCA(n_components=len(self.SC))
        pca.fit(df)
        explained_variance = np.array(pca.explained_variance_)

        return (np.sum(explained_variance)**2)/np.sum(explained_variance*explained_variance)

    def boldfitAR1(self):
        """
        Computes the ACF value at time lag = 1.

        Returns
        -------
        res: numpy array of length # ROI
            ACF values at time lag = 1 for each ROI.
        """
        res = np.zeros(self.bold.shape[0])

        for i in range(self.bold.shape[0]):
            res[i] = np.corrcoef(self.bold[i,:-1], self.bold[i,1:])[0,1]

        return res

    def boldfitARn(self, n):
        """
        Computes the ACF value at time lag = n.

        Returns
        -------
        res: numpy array
            ACF values at time lag = n for each ROI.
        """
        res = np.zeros(self.bold.shape[0])

        for i in range(self.bold.shape[0]):
            res[i] = np.corrcoef(self.bold[i,:-n], self.bold[i,n:])[0,1]

        return res

    def louvain_Q(self):
        """
        Compute modules and segregation
        """

        fc = self.simFC
        fc[np.where(fc<0)] = 0
        labels,Q = community_louvain(fc, gamma=2, ci=None, B='modularity', seed=None)

        return labels,Q

    def mean_BA(self):
        """
        compute mean participation coefficient
        """

        fc = self.simFC
        fc[np.where(fc<0)] = 0
        _,Q = self.louvain_Q()
        ci = [2,6,6,6,5,5,5,6,1,1,1,1,1,1,1,4,2,4,4,4,2,3,5,6,6,6,2,2,2,3,3,6,1,6,3,3,3,5]  # fulcher PNAS paper
        BA = participation_coef(fc, ci=ci, degree='undirected')

        return BA.mean(), Q  # integration, segregation







