import pandas as pd
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt

a = pd.read_csv('4.csv')
b = list(a.MKS)
c = list(a.TSCO)
d = list(a.SBRY)
e = list(a.HFD)
b1 ,c1,d1 = np.zeros((520,)),np.zeros((520,)),np.zeros((520,))

for i in range(1,521):
    b1[i-1] = b[i] - b[i-1]#np.log(b[i]/b[i-1]) 
    
for i in range(1,521):
    c1[i-1] =   c[i] - c[i-1] #np.log(c[i]/c[i-1]) 
    
for i in range(1,521):
    d1[i-1] =   d[i] - d[i-1]#np.log(d[i]/d[i-1])

bm, bv = np.average(b1), np.var(b1)
cm, cv = np.average(c1), np.var(c1)
dm, dv = np.average(d1), np.var(d1)

#1
T  = 2
interval =504
I = 5000
dt = T/interval
X = np.zeros((interval + 1,I))
gamma1 = np.sqrt(ev)
sigma1 = gamma1
X[0] = em
for i in range(1,interval+1):
    
     X[i] = X[i-1]* np.exp( ( (sigma1 **2*gamma1**2/(((i-1)*dt) *gamma1**2 + sigma1 **2)) - 0.5 * (sigma1 *gamma1**2/(((i-1)*dt) *gamma1**2 + sigma1 **2 ))**2) * dt
                 + (sigma1 *gamma1**2/(((i-1)*dt)* gamma1**2 + sigma1 **2))*np.sqrt(dt)*np.random.standard_normal(I)) 
                
g= list(a.HFD)[0] * sum(sum(np.exp(X)) * dt)/I   

#2
#e1 = np.zeros((260,))

for i in range(1,261):
    e1[i-1] =  e[i] - e[i-1]#np.log(d[i]/d[i-1])
    
em1,ev1 = np.average(e1), np.var(e1) 
X = np.zeros((interval+ 1,I))


gamma1 = np.sqrt(ev)
sigma1 = np.sqrt(ev1)
X[0] = em1
for i in range(1,interval+1):
    
     X[i] = X[i-1]* np.exp( ( (sigma1 **2*gamma1**2/(((i)*dt) *gamma1**2 + sigma1 **2)) - 0.5 * (sigma1 *gamma1**2/(((i)*dt) *gamma1**2 + sigma1 **2 ))**2) * dt
                 + (sigma1 *gamma1**2/(((i)*dt)* gamma1**2 + sigma1 **2))*np.sqrt(dt)*np.random.standard_normal(I)) 

g= list(a.HFD)[0] * sum(sum(np.exp(X)) * (dt - 1/interval))/I  

plt.figure(figsize = (7,7))
plt.plot( e,lw = 0.7,color ='black',label = 'HFD')
plt.axhline(165.12994411218912,linestyle = '--',linewidth = 0.7, color = 'red',label = 'S_star = 165.13')
plt.ylabel('Stock price')
plt.xlabel('Time interval')
plt.legend()