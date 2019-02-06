import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

def choice():
    for K in range(n):
          x[K] =  np.random.beta(s[K] + 1, f[K] + 1)
    return np.argmax(x)

np.set_printoptions(precision = 2)
        
n = 2
t = 100000
T = 100000
u = 995
v = 1000
delta = 0.001

regts = np.zeros(v - u)
regtserr = np.zeros(v - u)
regrts = np.zeros(v - u)
regrtserr = np.zeros(v - u)
regurts = np.zeros(v - u)
regurtserr = np.zeros(v - u)

trsts = np.zeros(v - u)
trstserr = np.zeros(v - u)
trsrts = np.zeros(v - u)
trsrtserr = np.zeros(v - u)
trsurts = np.zeros(v - u)
trsurtserr = np.zeros(v - u)

for y in range(u, v):
    qt = np.array([delta + y * (1 - delta) / (v - 1), y * (1 - delta) / (v - 1)])

    trj = np.zeros(t)
    ar = np.zeros(t)
    reg = 0
    regt = np.zeros(t)

    for i in range(0, t):
        f = np.zeros(n)
        s = np.zeros(n)
        x = np.zeros(n)    
        for j in range(0, T):

            d = choice()

    ##        if i == 99 and j > 9900:
    ##            print d, " ",


            if qt[d] > np.random.random():
                r = 1
                s[d] = s[d] + 1
            else : 
                r = 0
                f[d] = f[d] + 1
                
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

        f = np.zeros(n)
        s = np.zeros(n)
        x = np.zeros(n)

        r = 0
        for j in range(0, T):
            if r == 0:
                d = choice()

    ##        if i == 99 and j > 9900:
    ##            print d, " ",


            if qt[d] > np.random.random():
                r = 1
                s[d] = s[d] + 1
            else : 
                r = 0
                f[d] = f[d] + 1
                
            trj[i] = trj[i] + r

            regt[i] = regt[i] + max(qt) - qt[d]

    regrts[y - u] = np.mean(regt)
    regrtserr[y - u] = np.std(regt) / math.sqrt(t)
    
    trsrts[y - u] = np.mean(trj)
    trsrtserr[y - u] = np.std(trj) / math.sqrt(t)

    trj = np.zeros(t)
    ar = np.zeros(t)
    reg = 0
    regt = np.zeros(t)

    for i in range(0, t):

        f = np.zeros(n)
        s = np.zeros(n)
        x = np.zeros(n)
        U = np.ones(n)

        r = 0
        pers = 0
        
        for j in range(0, T):
            if r == 0 or pers >= U[d]:
                if pers >= U[d]:
                    U[d] = U[d] + 1
                pers = 0
                d = choice()
##            if y == u:
##                print(r, " ", d, " ", U[d], " ", pers)
            

    ##        if i == 99 and j > 9900:
    ##            print d, " ",


            if qt[d] > np.random.random():
                r = 1
                s[d] = s[d] + 1
                pers = pers + 1
            else : 
                r = 0
                f[d] = f[d] + 1
                
            trj[i] = trj[i] + r

            regt[i] = regt[i] + max(qt) - qt[d]

    regurts[y - u] = np.mean(regt)
    regurtserr[y - u] = np.std(regt) / math.sqrt(t)
    
    trsurts[y - u] = np.mean(trj)
    trsurtserr[y - u] = np.std(trj) / math.sqrt(t)


    print(y)    

print(delta)

for y in range(u, v):
    print(delta + y * (1 - delta) / (v - 1), " ", y * (1 - delta) / (v - 1), ": ", end = '')
    print(trsts[y - u], "+-", trstserr[y - u], " ", trsrts[y - u], "+-", trsrtserr[y - u], trsurts[y - u], "+-", trsurtserr[y - u])

plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), regts, regtserr, linestyle = 'None', marker = '^', label = 'reg')
plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), regrts, regrtserr, linestyle = 'None', marker = '^', label = 'rept')
plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), regurts, regurtserr, linestyle = 'None', marker = '^', label = 'urept')
plt.legend(loc = 'upper left')
plt.savefig('a.png')
plt.show()


##plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), trsts, trstserr, linestyle = 'None', marker = '^', label = 'reg')
##plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), trsrts, trsrtserr, linestyle = 'None', marker = '^', label = 'rept')
##plt.errorbar(np.arange(u, v) * (1 - delta) / (v - 1), trsurts, trsurtserr, linestyle = 'None', marker = '^', label = 'urept')
##plt.legend(loc = 'upper left')
##plt.show()
