import numpy as np  
import pandas as pd  
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns
import math

S0 = 30
sigma = 0.3 
T = 2.0
I = 1
I2 = 1
interval = 504
dt = T/interval
h =  0.5
l = - 0.5 
omega  = 1/0.3
theta = 1
epsi = 0.5 * (- sigma**2)
beta = 0.09
mu = -0.41

S = np.zeros((interval + 1,I))
B = np.zeros((interval + 1,I))
C=  np.zeros((interval + 1,I))
D=  np.zeros((interval + 1,I))
S[0] = S0 
B[0],C[0],D[0] = 25,22,18

for t in range(1,interval+1):

    S[t] = S[t-1] *np.exp(-0.41 * dt + 0.3 *np.sqrt(dt)*np.random.standard_normal(I))
    B[t] = B[t-1]* np.exp(epsi * dt) *(np.exp(0.1 * t * dt))**0.09


for t in range(1,interval+1):
    C[t] = C[t-1]* np.exp(-0.18 * dt) *(np.exp(0.1 * t * dt))**0.16


for t in range(1,interval+1):
    D[t] = D[t-1]* np.exp(-0.405 * dt) *(np.exp(0.1 * t * dt))**0.25
    
    
plt.figure(figsize = (6.5,6.5))
plt.plot( S[0:101],lw = 0.7,color ='black',label = 'Simulated path')
plt.plot( B[0:101],lw = 0.7,color ='black',ls = '--',label = 'Sigma = 0.3')
plt.plot( C[0:101],lw = 0.7,color ='black',ls = '-.',label = 'Sigma = 0.4')
plt.plot( D[0:101],lw = 0.7,color ='black',ls = ':',label = 'Sigma = 0.5')
plt.xlabel('Time interval')
plt.ylabel('Asset price')
plt.legend()
