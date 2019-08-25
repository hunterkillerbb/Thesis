import numpy as np  
import pandas as pd  
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns
import math


S0 = 30
x1 = -1
sigma1 = 0.3 
gamma1 = 0.5
T = 2.0
I = 5000
interval = 504
dt = T/interval
h =  0.5
l = - 0.5 
N =  1

S = np.zeros((61,))
X = np.zeros((interval + 1,I))
X[0] = x1


for sigma1 in np.arange(1,4.01,0.05):
    g = 0
    print(sigma1)
    for k in range(N):
        
        for i in range(1,interval+1):

            X[i] = X[i-1]*np.exp( ((h-X[i-1])*(X[i-1]-l) - 0.5 * (1/sigma1 * (h-X[i-1])*(X[i-1]-l))**2 ) * dt
                 + 1/sigma1 * (h-X[i-1])*(X[i-1]-l) *np.sqrt(dt)*np.random.standard_normal(I))
        
            s = sum(sum(np.exp(X)) * dt / I)   
            
        g += s

    S[int((sigma1-1.0)/0.05)] = S0 * g/N 



sig = np.linspace(1,4,61)

plt.figure(figsize = (6.5,6.5))
plt.plot( sig , S,lw = 0.7,color ='black',label = 'Value function')
plt.ylabel('V_hat')
plt.xlabel('Volatility')
plt.legend()
plt.annotate('',xytext=(2.0,30.75),xy=(3.0,30.25),arrowprops=dict(arrowstyle='->'))

