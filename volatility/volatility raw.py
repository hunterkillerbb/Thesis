
import numpy as np  
import pandas as pd  
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns
import math


S0 = 30
x1 = 0.3
sigma1 = 0.3 
gamma1 = 0.5
T = 2.0
I = 5000
interval = 504
dt = T/interval
h =  0.5
l = -0.5

def dps1(x,h,l,sigma1):
    if sigma1 == 0:
        return math.inf
    else:
        return 1/sigma1 * (h-x)*(x-l)
    
def dps2(gamma, sigma,t):
    return (sigma * gamma *gamma)/(sigma * sigma + t*gamma *gamma)


def X1(t, X0,dt,sigma1):
    X = np.exp( ( (h-X0)*(X0-l)- 0.5 * dps1(X0,h,l,sigma1)**2) * dt+ dps1(X0,h,l,sigma1)*np.sqrt(t+dt)*np.random.standard_normal(1))

    return X

def X2(t,dt,sigma1,gamma1):
    X = np.exp( (sigma1 * dps2(gamma1, sigma1 ,t) - 0.5 * dps2(gamma1, sigma1,t)**2) * dt+ dps2(gamma1, sigma1,t)*np.sqrt(t+dt)*np.random.standard_normal(1)) 
    return X


S = np.zeros((61,))
F = np.zeros((61,))

for sigma1 in np.arange(0.2,0.8,60):
    g = 0 
    for i in range(I):
            if i < I+1:
                g += np.exp(X1(0,x1,dt,sigma1)*T)
            else:
                break
            #x1 = X1(0,x1,j) 
            S[int((sigma1-0.2)/0.01)] = S0 * g/I 
           
for sigma1 in np.arange(0.2,0.8,60):
    g = 0 
    for i in range(I):
            if i < I+1:
                g += np.exp(X2(0,dt,sigma1,gamma1)*T)   
            else:
                break
            #x1 = X1(0,x1,j) 
            F[int((sigma1-0.2)/0.01)] = S0 * g/I 
   
sig = np.linspace(0.2,0.8,61)


plt.figure(figsize = (6.5,6.5))
plt.plot( sig , S,lw = 0.7,color ='black',linestyle ='--')
plt.plot( sig , F,lw = 0.7,color ='black' )
plt.ylabel('value function w.r.t. volatility')
plt.title('')
plt.show()




plt.annotate('vol = 1',xy=(175,10),xytext=(185,15),arrowprops=dict(arrowstyle='->'))
plt.annotate('vol = 0.3',xy=(175,53),xytext=(165,60),arrowprops=dict(arrowstyle='->'))
plt.annotate('Time Interval',xy=(102,-3) ,xytext=(102,-3))




      
    



