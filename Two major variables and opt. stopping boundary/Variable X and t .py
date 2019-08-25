
import numpy as np  
import pandas as pd  
from scipy.stats import norm 
import matplotlib.pyplot as plt
import seaborn as sns

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


S = np.zeros((5,5))
F = np.zeros((5,5))
X = np.zeros((interval + 1,I))
Y = np.zeros((interval + 1,I))

print(1)
for q in np.arange(-1,1.1,0.5):
    print(q)
    for j in np.arange(0.4,2.1,0.4):
           print(j)
           for i in range(1,interval + 1):
                X[0] = q 
                X[i] = X[i-1]*np.exp( ((h-X[i-1])*(X[i-1]-l) - 0.5 * (1/sigma1 * (h-X[i-1])*(X[i-1]-l))**2 ) * dt
                 + 1/sigma1 * (h-X[i-1])*(X[i-1]-l) *np.sqrt(dt)*np.random.standard_normal(I))
        
                g = sum(sum(np.exp(X)) * (dt - j/interval))/I   

           S[int((q + 1)/0.5)][int((j-0.4)/0.4)]  = S0 * g 
        
print(2)
for q in np.arange(-1,1.1,0.5):
    print(q)
    for j in np.arange(0.4,2.1,0.4):
        print(j)
        for i in range(1,interval + 1):
              Y[0] = q
              Y[i] = Y[i-1]* np.exp( ( (sigma1 **2*gamma1**2/((j+i*dt) *gamma1**2 + sigma1 **2)) - 0.5 * (sigma1 *gamma1**2/((j+i*dt) *gamma1**2 + sigma1 **2 ))**2) * dt
                 + (sigma1 *gamma1**2/((j+i*dt)* gamma1**2 + sigma1 **2))*np.sqrt(dt)*np.random.standard_normal(I)) 
                
              g= sum(sum(np.exp(Y)) * (dt - j/interval))/I   
                
        F[int((q + 1)/0.5)][int((j -0.4)/0.4)] = S0 *g

S = S.tolist()
S_df = pd.DataFrame(S)
writer = pd.ExcelWriter('FULL GRAPH(1).xlsx')
S_df.to_excel(writer,'2-POINT') # 
writer.save()

F = F.tolist()
F_df = pd.DataFrame(F)
writer = pd.ExcelWriter('FULL GRAPH(2).xlsx')
F_df.to_excel(writer,'NORMAL') # 
writer.save()
