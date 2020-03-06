import numpy as np
import scipy.optimize
import math
import matplotlib.pyplot as plt
from random import randint

def delta(p, q):
    if q >= 1:
        return float('inf') + q

    if p == 0:
        return math.log(1 / (1 - q))
    elif p == 1:
        return math.log(1 / q)

    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def f(q):
    return u[d] * delta(trn[d] / u[d], q) - math.log(sum(u))
    
def choice():
    return np.argmax(qmax)


n = 20
T = 1000
t = 1000

trj = np.zeros(t)
ar = np.zeros(t)
regt = np.zeros(t)


for i in range(0, T):
    qt = np.random.uniform(0, 1, n)
    #q = np.zeros(n)
    u = np.zeros(n)
    trn = np.zeros(n)
    #alpha = np.ones(n)
    qmax = np.zeros(n)

    for j in range(0, n):
        d = j
        
        if qt[d] > np.random.random():
            r = 1
        else : 
            r = 0
        
        u[d] = 1
        trn[d] = r
        qmax[d] = min(1, scipy.optimize.newton(f, trn[d]/u[d], maxiter = 100))

        regt[j] = regt[j] + max(qt) - qt[d]
        #print d, " ", qmax[d], " ", f(qmax[d])           
    for j in range(n, t):
        d = choice()

        if qt[d] > np.random.random():
            r = 1
        else : 
            r = 0
            
        u[d] = u[d] + 1
        trn[d] = trn[d] + r
        qmax[d] = min(1, scipy.optimize.newton(f, trn[d]/u[d], maxiter = 100))

        #print d, " ", qmax[d], " ", f(qmax[d])               
        #q[d] = q[d] + (r - q[d]) * alpha[d]
        trj[j] = trj[j] + r
        #alpha[d] = alpha[d]/(alpha[d] + 1)

        regt[j] = regt[j] + max(qt) - qt[d]

ar = trj/T  

reg = np.cumsum(regt)

#print (reg[t - 1]/T)

#plt.plot(reg, label = t)
plt.plot(reg/T, label = t)

plt.legend(loc = 'upper left')
#plt.show()
