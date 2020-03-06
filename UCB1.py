import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from random import randint

def choice():
    if np.sum(u) > 0 :
        return np.argmax(q + np.sqrt(2 * math.log(np.sum(u)) / u))
    else :
        return randint(0, n - 1)

n = 20
T = 1000
t = 1000

trj = np.zeros(t)
ar = np.zeros(t)
reg = 0
regt = np.zeros(t)

for i in range(0, T):
    qt = np.random.uniform(0, 1, n)
    q = np.zeros(n)
    u = np.zeros(n)
    alpha = np.ones(n)      
    for j in range(0, t):

        d = choice()

##        if i == 99 and j > 9900:
##            print d, " ",
    
        u[d] = u[d] + 1

        if qt[d] > np.random.random():
            r = 1
        else : 
            r = 0
            
        q[d] = q[d] + (r - q[d]) * alpha[d]
        trj[j] = trj[j] + r
        alpha[d] = alpha[d]/(alpha[d] + 1)

        regt[j] = regt[j] + max(qt) - qt[d]

ar = trj/T  

reg = np.cumsum(regt)/T

print(reg[t - 1])
df = pd.DataFrame(reg)
df.to_csv("regret_UCB1.csv", header = None, index = None)
#plt.plot(reg, label = t)
plt.plot(reg, label = t)
plt.plot(np.log(1 + np.arange(t))*np.sum(8 / ((1 + np.arange(n)) / float(n))))
plt.legend(loc = 'upper left')
plt.show()
