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
l = - 0.5 
N =  1

S = np.zeros((31,5))
X = np.zeros((interval + 1,I))
X[0] = x1

for t in np.arange(0,2.1,0.5):
    print(t)
    for gamma1 in np.arange(0.1,1.61,0.05):
        g = 0
        print(gamma1)
        for k in range(N):
            
            for i in range(1,interval+1):
                
                X[i] = X[i-1]* np.exp( ( (sigma1 **2*gamma1**2/(t*gamma1**2 + sigma1 **2)) - 0.5 * (sigma1 *gamma1**2/(t*gamma1**2 + sigma1 **2))**2) * dt
                 + (sigma1 *gamma1**2/(t* gamma1**2 + sigma1 **2))*np.sqrt(dt)*np.random.standard_normal(I)) 
                
                s = sum(sum(np.exp(X)) * (dt - t/interval)) / I   
                
                
            g += s
                
        S[int((gamma1-0.1)/0.05)][int(t/0.5)] = S0 * g/N 
        
        

dis = np.linspace(0.1,1.6,31)

plt.figure(figsize = (6.5,6.5))
plt.plot( dis , S[:,0], lw = 0.7,color ='black',label = 'Value functions')
plt.plot( dis , S[:,1], lw = 0.7,color ='black',label = 'Value functions')
plt.plot( dis , S[:,2], lw = 0.7,color ='black',label = 'Value functions')
plt.plot( dis , S[:,3], lw = 0.7,color ='black',label = 'Value functions')
plt.plot( dis , S[:,4], lw = 0.7,color ='black',label = 'Value functions')
plt.ylabel('V_hat')
plt.xlabel('gamma')
plt.legend()

S = S.tolist()
S_df = pd.DataFrame(S)
writer = pd.ExcelWriter('Initial prior normal.xlsx')
S_df.to_excel(writer,'NORMAL') 
writer.save()
