import numpy as np  
import pandas as pd  
from scipy.stats import norm 
import matplotlib.pyplot as plt
import math


S0 = 30
x1 = 0.3
sigma1 = 0.3 
gamma1 = 0.5
T = 2
I = 5000
interval = 504
dt = T/interval
h =   0.5
l = - 0.5
N = 1 
F = np.zeros((61,))
X = np.zeros((interval + 1,I))
X[0] = x1


for sigma1 in np.arange(1 ,4.01,0.05):
    g = 0
    print(sigma1)
    for k in range(N):
        
        for i in range(1,interval+1):

            X[i] = X[i-1]* np.exp( ( (sigma1 **2*gamma1**2/(gamma1**2 + sigma1 **2)) - 0.5 * (sigma1 *gamma1**2/(gamma1**2 + sigma1 **2))**2) * dt
            + (sigma1 *gamma1**2/(gamma1**2 + sigma1 **2))*np.sqrt(dt)*np.random.standard_normal(I)) 
           
            s = sum(sum(np.exp(X)) * (dt - 1/interval)) / I   
            
        g += s

    F[int((sigma1-1)/0.05)] = S0 * g/N 



sig = np.linspace(1,4,61)


plt.figure(figsize = (6.5,6.5))
plt.plot( sig , F,lw = 0.7,color ='black',label = 'Value function')
plt.ylabel('V_hat')
plt.xlabel('Volatility')
plt.legend()
plt.annotate('',xytext=(2, 44),xy=(3,44.2),arrowprops=dict(arrowstyle='->'))