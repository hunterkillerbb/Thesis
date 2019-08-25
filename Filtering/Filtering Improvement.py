
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
T = 2
I = 5000
interval = 504
dt = T/interval
h =   0.5
l = - 0.5
N = 1 
S = np.zeros((61,))
F = np.zeros((61,))
X = np.zeros((interval + 1,I))
X[0] = x1


for sigma1 in np.arange(1 ,4.01,0.05):
    g = 0
    print(sigma1)
    for k in range(N):
        
        for i in range(1,interval+1):
            
            
            
            
            X[i] = X[i-1]* np.exp( ( (sigma1 **2*gamma1**2/((1+i*dt)*gamma1**2 + sigma1 **2)) - 0.5 * (sigma1 *gamma1**2/((1+i*dt)*gamma1**2 + sigma1 **2))**2) * dt
            + (sigma1 *gamma1**2/((1+i*dt)*gamma1**2 + sigma1 **2))*np.sqrt(dt)*np.random.standard_normal(I)) 
           
            s = sum(sum(np.exp(X)) * (dt - 1/interval)) / I  
            
            
        g += s

    F[int((sigma1-1)/0.05)] = S0 * g/N 


for sigma1 in np.arange(1,4.01,0.05):
    g = 0
    print(sigma1)
    for k in range(N):
        
        for i in range(1,interval+1):
            
            
             X[i] = X[i-1]*np.exp( ((h-X[i-1])*(X[i-1]-l) - 0.5 * (1/sigma1 * (h-X[i-1])*(X[i-1]-l))**2 ) * dt
                 + 1/sigma1 * (h-X[i-1])*(X[i-1]-l) *np.sqrt(dt)*np.random.standard_normal(I))
        
             s = sum(sum(np.exp(X)) * (dt-1/interval)  / I) 
            
        g += s

    S[int((sigma1-1.0)/0.05)] = S0 * g/N 

S1 =  S0 * np.exp(0.3 * (T-1)) * np.ones((61,))

S_hat = (S- S1)/S1 *100
F_hat = (F - S1)/S1 *100
#S1_hat = np.zeros((61,))

sig = np.linspace(1,4,61)

plt.figure(figsize = (6.5,6.5))
plt.plot( sig , S_hat,lw = 0.7,color ='black',label = 'Filtering and 2-point', ls = '--')
plt.plot( sig , F_hat,lw = 0.7,color ='black',label = 'Filtering and normal')
#plt.plot(sig,S1_hat , lw = 0.7,color = 'black', ls = '-.',label = 'No filtering' )
plt.ylabel('Improvement(%)')
plt.xlabel('Volatility')
plt.legend()
plt.annotate('',xytext=(2, 8),xy=(3,8.5),arrowprops=dict(arrowstyle='->'))
plt.annotate('',xytext=(2, 5.5),xy=(3,5.5),arrowprops=dict(arrowstyle='->'))


S = S.tolist()
S_df = pd.DataFrame(S)
writer = pd.ExcelWriter('filtering 2-point.xlsx')
S_df.to_excel(writer,'2-POINT') 
writer.save()

F = F.tolist()
F_df = pd.DataFrame(F)
writer = pd.ExcelWriter('filtering normal.xlsx')
F_df.to_excel(writer,'NORMAL') 
writer.save()


