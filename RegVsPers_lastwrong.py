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

np.set_printoptions(precision = 2)
        
n = 2
t = 100
T = 10000
u = 9
v = 10
delta = 0.2


lastwrong = np.zeros((2, t))

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
    qt = [0.99, 0.89]
    
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

            if np.argmax(s/(s+f)) != np.argmax(qt):
                lastwrong[0][i] = j

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
            if np.argmax(s/(s+f)) != np.argmax(qt):
                lastwrong[1][i] = j


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


    print(y)    

print(delta)

for y in range(u, v):
    print(delta + y * (1 - delta) / (v - 1), " ", y * (1 - delta) / (v - 1), ": ", end = '')
    print(regts[y - u], "+-", regtserr[y - u], " ", regrts[y - u], "+-", regrtserr[y - u], "  ", trsts[y - u], "+-", trstserr[y - u], " ", trsrts[y - u], "+-", trsrtserr[y - u])

print(lastwrong)
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
##
##plt.errorbar(qt[1], regts, regtserr, linestyle = 'None', marker = '^', label = 'reg')
##plt.errorbar(qt[1], regrts, regrtserr, linestyle = 'None', marker = '^', label = 'rept')
##plt.legend(loc = 'upper left')
##
##plt.savefig('RegVsPers' + str(qt[0]) + '_' + str(qt[1]) + '_' + str(T) + '.png')
##plt.show()

plt.hist(lastwrong[0])
plt.title('Last wrong arm identified for TS')
plt.savefig('last_wrong_arm_ts_' + str(qt[0]) + '_' + str(qt[1]) + '_' + str(t) + '.png')
plt.clf()

plt.hist(lastwrong[1])
plt.title('Last wrong arm identified for TS Persistence')
plt.savefig('last_wrong_arm_ts_pers_' + str(qt[0]) + '_' + str(qt[1]) + '_' +  str(t) + '.png')

