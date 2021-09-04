#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Builds the base class for large scale simulation"""
__author__ = "Daming Li"
__email__ = "daming.li@yale.edu"


import numpy as np
import math
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize
import networkx as nx
from community import community_louvain
from tools import *

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
        range_t = np.arange(dynamics.shape[1])*dt

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
        singular = np.array(pca.singular_values_)
        
        return (np.sum(singular)**2)/np.sum(singular*singular)

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
            res[i] = acf(self.bold[i,:], unbiased=True)[1]

        return res

    def louvainQ(self, thr=-1):
        """
        Computes Louvain's Q
        
        Parameters
        ----------
        thr:
            a non-negative threshold for cutoff of the FC matrix
        Returns
        -------
        Q:
            Louvain's Q measure for modularity
        """
        
        #W = np.clip(self.simFC, a_min=thr, a_max=None)
        W = self.simFC
        if thr >= 0:
            W[W<thr] = 0
        G = nx.from_numpy_array(W)
        louvain_partition = community_louvain.best_partition(G, weight='weight')
        Q = community_louvain.modularity(louvain_partition, G, weight='weight')
        
        return Q


    





