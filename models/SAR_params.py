import numpy as np

def F(x,a,theta): 
    '''
    Population activation function.

    Expects:
    x     : the population input
    a     : the gain of the function
    theta : the threshold of the function
    
    Returns:
    the population activation response F(x) for input x
    '''
    # add the expression of f = F(x)
    x = x.clip(min=0.0)
    f = ((1+np.exp(-a*(x-theta)))**-1)

    return f

tauE = 20.0/1000  #s
tauOU = 1./1000  #s
mu = 0.0
dt = 0.5/1000  # s