import numpy as np
import math

def subdiag(x):
    """
    Returns the upper diagonal of the matrix 
    """
    N = x.shape[0]
    return x[np.triu_indices(N, k=1)]

def gbc(FC):
    """
    Returns the global brain connectivity
    """
    fc = FC.copy()
    idiag = np.where(np.identity(fc.shape[0]))
    fc[idiag] = 0
    gbc = fc.sum(axis=1)/(fc.shape[0]-1)
    return gbc

def fbold(t, p=2., taub=1.25, o=2.25):
    """fbold(t) kernel for vector t"""
    kernel  = np.zeros_like(t)
    idx = t >= o
    kernel[idx] = ((t[idx] - o)/taub)**(p-1) / math.factorial(p-1) * np.exp(-(t[idx] - o)/taub)
    
    return kernel