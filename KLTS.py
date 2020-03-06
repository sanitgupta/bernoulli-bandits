import numpy as np
import math
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint
from scipy.optimize import brenth

def kl(p, q):
    if q == p:
        return 0
    elif q == 1 or q == 0:
        return float('inf')
    elif p == 0:
        return math.log(1/(1 - q))
    elif p == 1:
        return math.log(1 / q)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

def fun(q, K):
    return (S[K]+F[K]) * kl(s[K] / (S[K]+F[K]), q) - math.log(n*H/C)


def ts_choice():
    for K in range(n):
          x[K] =  np.random.beta(s[K] + 1, f[K] + 1)
    return np.argmax(x)

np.set_printoptions(precision = 2)
        

t = 20
T = 1000
H = 1000
C = 1

regts = np.zeros(T)
regtserr = np.zeros(T)
regrts = np.zeros(T)
regrtserr = np.zeros(T)
regurts = np.zeros(T)
regurtserr = np.zeros(T)

trsts = np.zeros(T)
trstserr = np.zeros(T)
trsrts = np.zeros(T)
trsrtserr = np.zeros(T)
trsurts = np.zeros(T)
trsurtserr = np.zeros(T)


n = 20
qt = np.append(0.6, 0.49 * np.ones(n - 1))
#qt = np.arange(0, n) * 1 / n + 1 / (2 * n)
#qt = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
print(qt)
n = len(qt)


trj = np.zeros(t)
ar = np.zeros(t)
reg = 0
regt = np.zeros((t,T))

for i in range(0, t):

    f = np.zeros(n)
    s = np.zeros(n)
    F = np.zeros(n)
    S = np.zeros(n)
    x = np.zeros(n)

    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for j in range(0, T):
        
        d = ts_choice()
        if qt[d] > np.random.random():
            r = 1
            s[d] = s[d] + 1
            S[d] = S[d] + 1
        else : 
            r = 0
            f[d] = f[d] + 1
            F[d] = F[d] + 1
        trj[i] = trj[i] + r

        regt[i][int(np.sum(S+F)) - 1] = max(qt) - qt[d]

        if np.sum(S+F) >= H:
            break

        if(np.sum(S+F>0)>=n):
            mean = S/(S+F)
            #bnd = np.sqrt( (np.log(c*n/H)) #+ np.log(np.log(k1*n*np.sum(s+f)**alpha/delta))) /(2*U))
            for K in range(n):
                if fun(1, K) <= 0:
                    upper[K] = 1
                else:
                    upper[K] = brenth(fun, S[K]/(S[K]+F[K]), 1, args = (K,))

                if fun(0, K) <= 0:
                    lower[K] = 0
                else:
                    lower[K] = brenth(fun, 0, S[K]/(S[K]+F[K]), args = (K,))


            upper = np.ma.array(upper, mask = False)
            
            d1 = np.argmax(mean)

            upper.mask[d1] = True
            
            d2 = np.argmax(upper)

            upper.mask[d2] = False

            if upper[d2] - lower[d1] < 0: #stopping condition
                if qt[d1] > np.random.random():
                    r = 1
                    S[d1] = S[d1] + 1
                else : 
                    r = 0
                    F[d1] = F[d1] + 1

                trj[i] = trj[i] + r

                regt[i][int(np.sum(S+F)) - 1] = max(qt) - qt[d1]

            if np.sum(S+F) >= H:
                break
            

        
regts = np.mean(np.cumsum(regt, axis = 1), axis = 0)
regtserr = np.std(np.cumsum(regt, axis = 1), axis = 0) / math.sqrt(t)

trsts = np.mean(trj)
trstserr = np.std(trj) / math.sqrt(t)

plt.plot(1 + np.arange(np.sum(S+F)), regts, label = 'TS')
plt.legend(loc = 'upper left')
plt.show()
