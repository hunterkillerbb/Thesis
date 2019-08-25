import numpy as np  
import pandas as pd  
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns
import math

S0 = 30
x1 = 0.3
sigma1 = 1 
gamma1 = 0.5
T = 2.0
I = 5000
interval = 504
dt = T/interval
h =  0.5
l = - 0.5 
N =  1

S = np.zeros((31,))
X = np.zeros((interval + 1,I))
X[0] = x1


for j in np.arange(0.1,1.61,0.05):
    g = 0
    print(j)
    for k in range(N):
        
        for i in range(1,interval+1):
            h = j 
            l = j - 1
            X[i] = X[i-1]*np.exp( ((h-X[i-1])*(X[i-1]-l) - 0.5 * (1/sigma1 * (h-X[i-1])*(X[i-1]-l))**2 ) * dt
                 + 1/sigma1 * (h-X[i-1])*(X[i-1]-l) *np.sqrt(dt)*np.random.standard_normal(I))
        
            s = sum(sum(np.exp(X)) * dt / I)   
            
        g += s

    S[int((j-0.1)/0.05)] = S0 * g/N 



dis = np.linspace(0.1,1.6,31)

plt.figure(figsize = (6.5,6.5))
plt.plot( dis , S,lw = 0.7,color ='black',label = 'Value function')
plt.ylabel('V_hat')
plt.xlabel('h')
plt.legend()
plt.vlines(0.85,74.2,S[15],linestyles = '--',linewidth = 1.1)
plt.annotate('(0.85, 88.76)',xytext=(1.2,86),xy=(0.85, 88.76),arrowprops=dict(arrowstyle='-'))
plt.annotate('',xytext=(0.85,74.2),xy=(0.85, 73),arrowprops=dict(arrowstyle='->'))