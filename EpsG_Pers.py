import numpy as np
import math
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint

def choice():
    for K in range(n):
          x[K] =  np.random.beta(s[K] + 1, f[K] + 1)
    return np.argmax(x)

def ds_choice():
    samp = np.zeros(n)
    while len(np.where(samp == max(samp))[0]) > 1:
        for K in range(n):
            samp[K] =  np.random.choice(len(dist[K]), p = dist[K])
    return np.argmax(samp)

def eps_choice():
    if np.random.random() < eps:
        return np.random.randint(n)
    else:
        return np.argmax(mu)


def normalize(a, axis=-1, order=1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


np.set_printoptions(precision = 2)

eps = 0.05

        
n = 2
t = 100
T = 60000
u = 9
v = 10
delta = 0.2

regts = np.zeros(v - u)
regtserr = np.zeros(v - u)
regrts = np.zeros(v - u)
regrtserr = np.zeros(v - u)

trsts = np.zeros(v - u)
trstserr = np.zeros(v - u)
trsrts = np.zeros(v - u)
trsrtserr = np.zeros(v - u)

for y in range(u, v):
    qt = np.array([delta + y * (1 - delta) / (v - 1), y * (1 - delta) / (v - 1)])
    qt = [0.9, 0.5]
    
    trj = np.zeros(t)
    ar = np.zeros(t)
    reg = 0
    regt = np.zeros(t)

    for i in range(0, t):
        mu = np.zeros(n)
        alpha = np.ones(n)

        for j in range(0, n):

            d = j
            
            if qt[d] > np.random.random():
                r = 1
            else : 
                r = 0

            mu[d] = (1 - alpha[d]) * mu[d] + alpha[d] * r

            alpha[d] = alpha[d] / (alpha[d] + 1)
                
            trj[i] = trj[i] + r

            regt[i] = regt[i] + max(qt) - qt[d]

        for j in range(n, T):

            d = eps_choice()

            if qt[d] > np.random.random():
                r = 1
            else : 
                r = 0

            mu[d] = (1 - alpha[d]) * mu[d] + alpha[d] * r

            alpha[d] = alpha[d] / (alpha[d] + 1)
                
            trj[i] = trj[i] + r

            regt[i] = regt[i] + max(qt) - qt[d]

    regts[y - u] = np.mean(regt)
    regtserr[y - u] = np.std(regt) / math.sqrt(t)

    trsts[y - u] = np.mean(trj)
    trstserr[y - u] = np.std(trj) / math.sqrt(t)


    trj = np.zeros(t)
    ar = np.zeros(t)
    reg = 0
    regt = np.zeros(t)
    
    for i in range(0, t):

        mu = np.zeros(n)
        alpha = np.ones(n)
        
        for j in range(0, n):

            d = j
            
            if qt[d] > np.random.random():
                r = 1
            else : 
                r = 0

            mu[d] = (1 - alpha[d]) * mu[d] + alpha[d] * r

            alpha[d] = alpha[d] / (alpha[d] + 1)
                
            trj[i] = trj[i] + r

            regt[i] = regt[i] + max(qt) - qt[d]

        r = 0
        for j in range(n, T):
            if r == 0:
                d = eps_choice()

            if qt[d] > np.random.random():
                r = 1
            else : 
                r = 0

            mu[d] = (1 - alpha[d]) * mu[d] + alpha[d] * r

            alpha[d] = alpha[d] / (alpha[d] + 1)
                
            trj[i] = trj[i] + r

            regt[i] = regt[i] + max(qt) - qt[d]

    regrts[y - u] = np.mean(regt)
    regrtserr[y - u] = np.std(regt) / math.sqrt(t)
    
    trsrts[y - u] = np.mean(trj)
    trsrtserr[y - u] = np.std(trj) / math.sqrt(t)


    print(y)    

print(delta)

for y in range(u, v):
    print(delta + y * (1 - delta) / (v - 1), " ", y * (1 - delta) / (v - 1), ": ", end = '')
    print(regts[y - u], "+-", regtserr[y - u], " ", regrts[y - u], "+-", regrtserr[y - u], "  ", trsts[y - u], "+-", trstserr[y - u], " ", trsrts[y - u], "+-", trsrtserr[y - u])

##plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), regts, regtserr, linestyle = 'None', marker = '^', label = 'reg')
##plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), regrts, regrtserr, linestyle = 'None', marker = '^', label = 'rept')
##plt.legend(loc = 'upper left')
##
##plt.savefig('RegVsPers' + str(qt[0]) + '_' + str(qt[1]) + '_' + str(T) + '.png')
##plt.show()
##
##plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), trsts, trstserr, linestyle = 'None', marker = '^', label = 'reg')
##plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), trsrts, trsrtserr, linestyle = 'None', marker = '^', label = 'rept')
##plt.legend(loc = 'upper left')
##plt.show()

plt.errorbar(qt[1], regts, regtserr, linestyle = 'None', marker = '^', label = 'reg')
plt.errorbar(qt[1], regrts, regrtserr, linestyle = 'None', marker = '^', label = 'pers')
plt.legend(loc = 'upper left')

plt.savefig('EG_RegVsPers_' + str(qt[0]) + '_' + str(qt[1]) + '_' + str(T) + '.png')
plt.show()

