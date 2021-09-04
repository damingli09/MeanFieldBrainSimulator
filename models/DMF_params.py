import numpy as np

def F(I, g=270., b=108., d=0.154):
    """F(I) for vector I"""

    return (g*I - b + 1e-8)/(1.-np.exp(-d*(g*I - b + 1e-8)))

tauS = 50.0/1000
tauA = 500.0/1000  #+ np.random.rand()*0.01 - 0.005
saturation = 0.641  # gamma in the equation
JA = 0.0002*0
JN = 0.2609
dt = 1.0/1000  # s
resolution = int(1/dt)  # how many steps count for 1s
gainconst = 270.
tauOU = 30./1000  #s
mu = 0.0