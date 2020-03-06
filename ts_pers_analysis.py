import numpy as np
import math
import matplotlib.pyplot as plt
from math import factorial as fac
from fractions import Fraction

##Fraction('3.1415926535897932').limit_denominator(1000)


def ex(x, n):
    t = 1
    for i in range(n):
        t = t*x
    return t

#probability of pulling optimal arm in state (s, f, s2, f2)
def probOpt(s, f, s2, f2):
    if s < 0 or f < 0 or s2 < 0 or f2 < 0:
        return 0
    res = 0
    for i in range(f + 1):
        for j in range(f2 + 1):
            res = res + Fraction((-1)**(i+j), fac(f-i)*fac(f2-j)*fac(i)*fac(j)*(s2+j+1)*(s+s2+i+j+2))

    return Fraction(res * fac(s+f+1) * fac(s2+f2+1), fac(s) * fac(s2))

#probability of pulling optimal arm in state (s, f, s2, f2) : persistence
def probPOpt(s, f, s2, f2, p):
    if s < 0 or f < 0 or s2 < 0 or f2 < 0:
        return 0
    if p == 1:
        return 1
    elif p == 2:
        return 0
    res = 0
    for i in range(f + 1):
        for j in range(f2 + 1):
            res = res + Fraction((-1)**(i+j), fac(f-i)*fac(f2-j)*fac(i)*fac(j)*(s2+j+1)*(s+s2+i+j+2))

    return Fraction(res * fac(s+f+1) * fac(s2+f2+1), fac(s) * fac(s2))

#path component of probability of ending up in a given state
def K(s, f, s2, f2):
    
    if s == 0 and f == 0 and s2 == 0 and f2 == 0:
        return 1

    elif s == -1 or f == -1 or s2 == -1 or f2 == -1 :
        return 0

    else:
        return (K(s - 1, f, s2, f2) * probOpt(s - 1, f, s2, f2)
        +  K(s, f - 1, s2, f2) * probOpt(s, f - 1, s2, f2)
        +  K(s, f, s2 - 1, f2) * (1 - probOpt(s, f, s2 - 1, f2))
        +  K(s, f, s2, f2 - 1) * (1 - probOpt(s, f, s2, f2 - 1)))

#path component of probability of ending up in a given state : persistence
def KP(s, f, s2, f2, p):
    
    if s == 0 and f == 0 and s2 == 0 and f2 == 0:
        if p == 0:
            return 1
        else:
            return 0
        
    elif s == -1 or f == -1 or s2 == -1 or f2 == -1 :
        return 0
        
    elif p == 0:
        return (KP(s, f - 1, s2, f2, 0) * probPOpt(s, f - 1, s2, f2, 0)
        +  KP(s, f, s2, f2 - 1, 0) * (1 - probPOpt(s, f, s2, f2 - 1, 0))

        +  KP(s, f - 1, s2, f2, 1) * probPOpt(s, f - 1, s2, f2, 1)
        +  KP(s, f, s2, f2 - 1, 1) * (1 - probPOpt(s, f, s2, f2 - 1, 1))

        +  KP(s, f - 1, s2, f2, 2) * probPOpt(s, f - 1, s2, f2, 2)
        +  KP(s, f, s2, f2 - 1, 2) * (1 - probPOpt(s, f, s2, f2 - 1, 2)))

    elif p == 1:
        return (KP(s - 1, f, s2, f2, 0) * probPOpt(s - 1, f, s2, f2, 0)

        +  KP(s - 1, f, s2, f2, 1) * probPOpt(s - 1, f, s2, f2, 1)

        +  KP(s - 1, f, s2, f2, 2) * probPOpt(s - 1, f, s2, f2, 2))

    elif p == 2:
        return (KP(s, f, s2 - 1, f2, 0) * (1 - probPOpt(s, f, s2 - 1, f2, 0))

        +  KP(s, f, s2 - 1, f2, 1) * (1 - probPOpt(s, f, s2 - 1, f2, 1))

        +  KP(s, f, s2 - 1, f2, 2) * (1 - probPOpt(s, f, s2 - 1, f2, 2)))

#mean component of probability of ending up in a given state
def probMeanState(s, f, s2, f2):
    
    if s == 0 and f == 0 and s2 == 0 and f2 == 0:
        return np.ones((1,1))
    
    v = np.append(1, np.zeros(s + f + s2 + f2))
    vs = np.append(0, 1)
    vs = np.append(vs, np.zeros(s + f + s2 + f2 - 1))
    vf = np.append(1, - 1)
    vf = np.append(vf, np.zeros(s + f + s2 + f2 - 1))

    v2 = np.append(1, np.zeros(s + f + s2 + f2))
    vs2 = np.append(0, 1)
    vs2 = np.append(vs2, np.zeros(s + f + s2 + f2 - 1))
    vf2 = np.append(1, - 1)
    vf2 = np.append(vf2, np.zeros(s + f + s2 + f2 - 1))

    for t in range(s + f):
        if t < s:
            z = np.outer(v, vs)
        else:
            z = np.outer(v, vf)

        v = np.zeros(s + f + s2 + f2 + 1)
        for i in range(len(v)):
            for j in range(len(v)):
                if z[i][j] != 0:
                    v[i + j] += z[i][j]

    for t in range(s2 + f2):
        if t < s2:
            z2 = np.outer(v2, vs2)
        else:
            z2 = np.outer(v2, vf2)

        v2 = np.zeros(s + f + s2 + f2 + 1)
        for i in range(len(v2)):
            for j in range(len(v2)):
                if z2[i][j] != 0:
                    v2[i + j] += z2[i][j]

    return np.outer(v, v2).astype(int) + Fraction()

#probability of ending up in a given state
def probState(s, f, s2, f2):
    if s + f + s2 + f2 == 0:
        return 1
    return K(s, f, s2, f2) * probMeanState(s, f, s2, f2)

#components of probPState
def probPWiseState(s, f, s2, f2, p):
    return KP(s, f, s2, f2, p) * probMeanState(s, f, s2, f2)

#probability of ending up in a given state : persistence
def probPState(s, f, s2, f2):
    return probPWiseState(s, f, s2, f2, 0) + probPWiseState(s, f, s2, f2, 1) + probPWiseState(s, f, s2, f2, 2)

#expected reward in a given time horizon
def expR(T):
    p = np.zeros((T+1, T+1)).astype(int) + Fraction()
    for t in range(T):
        z = np.zeros((t+2,t+2)).astype(int) + Fraction()
        z[0, 1] = Fraction(1, 1)
        
        x = np.hstack((np.vstack([np.zeros(t+1).astype(int) + Fraction(), probOptN(t)]), np.zeros((t + 2, 1)).astype(int) + Fraction())) + z - np.hstack((np.zeros((t + 2, 1)).astype(int) + Fraction(),np.vstack([probOptN(t), np.zeros(t+1).astype(int) + Fraction()])))
        s = np.zeros((T+1, T+1)).astype(int) + Fraction() 
        s[:x.shape[0], :x.shape[1]] = x
        p = p + s
        
    return p + Fraction()

#expected reward in a given time horizon : persistence
def expPR(T):
    p = np.zeros((T+1, T+1)).astype(int) + Fraction()
    for t in range(T):
        z = np.zeros((t+2,t+2)).astype(int) + Fraction()
        z[0, 1] = Fraction(1, 1)
        
        x = np.hstack((np.vstack([np.zeros(t+1).astype(int) + Fraction(), probPOptN(t)]), np.zeros((t + 2, 1)).astype(int) + Fraction())) + z - np.hstack((np.zeros((t + 2, 1)).astype(int) + Fraction(),np.vstack([probOptN(t), np.zeros(t+1).astype(int) + Fraction()])))
        s = np.zeros((T+1, T+1)).astype(int) + Fraction() 
        s[:x.shape[0], :x.shape[1]] = x
        p = p + s
        
    return p + Fraction()

#numerical value of probState given mu, mu'
def probStateNUM(mu, mu2, s, f, s2, f2):
    x = probState(s,f,s2,f2)
    t = matNum(x, mu, mu2);

    return t
            
#probability of choosing optimal action at time n given thompson sampling was done from the base state
def probOptN(n):
    z = 0
    for s in range(n + 1):
        for f in range(n + 1 - s):
            for s2 in range(n + 1 - s - f):
                f2 = n - s - f - s2
                z = z + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)
    return z

def probPOptN(n):
    z = 0
    for s in range(n + 1):
        for f in range(n + 1 - s):
            for s2 in range(n + 1 - s - f):
                f2 = n - s - f - s2
                for p in range(3):
                    z = z + probPOpt(s, f, s2, f2, p) * probPWiseState(s, f, s2, f2, p)
    return z



def matNum(x, mu, mu2):
    t = 0
    for i in range(len(x)):
        for j in range(len(x)):
            t = t + ex(mu, i) * ex(mu2, j) * x[i, j]
    return t




def probOptNext(s, f, s2, f2):
    return probOpt(s, f, s2, f2) * (probMeanState(1, 0, 0, 0) * probOpt(s + 1, f, s2, f2) + probMeanState(0, 1, 0, 0) * probOpt(s, f + 1, s2, f2)) + (1 - probOpt(s, f, s2, f2)) * (probMeanState(0, 0, 1, 0) * probOpt(s, f, s2 + 1, f2) + probMeanState(0, 0, 0, 1) * probOpt(s, f, s2, f2 + 1))

def probOptAB(a, b):
    n = 0
    d = 0
    for s in range(a + 1):
        f = a - s
        for s2 in range(b + 1):
            f2 = b - s2
            n = n + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)
            d = d + probState(s, f, s2, f2) 
    return n, d

def probOptSF(S, F):
    n = 0
    d = 0
    for s in range(S + 1):
        s2 = S - s
        for f in range(F + 1):
            f2 = F - f
            n = n + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)
            d = d + probState(s, f, s2, f2) 
    return n, d

def probOptABU2(s, f, b):
    n = 0
    d = 0

    for s2 in range(b + 1):
        f2 = b - s2
        n = n + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)
        d = d + probState(s, f, s2, f2)
    return n, d

def probOptUAB(a, s2, f2):
    n = 0
    d = 0

    for s in range(a + 1):
        f = a - s
        n = n + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)
        d = d + probState(s, f, s2, f2)
    return n, d

def probOptABNUM(a, b, mu, mu2):
    n, d = probOptAB(a, b)
    t1 = matNum(n, mu, mu2)
    t2 = matNum(d, mu, mu2)
            
    return (t1/t2)

def probOptSFNUM(S, F, mu, mu2):
    n, d = probOptSF(S, F)
    t1 = matNum(n, mu, mu2)
    t2 = matNum(d, mu, mu2)
            
    return (t1/t2)

def probOptABU2NUM(a, b, u2, mu, mu2):
    n, d = probOptABU2(a, b, u2)
    t1 = matNum(n, mu, mu2)
    t2 = matNum(d, mu, mu2)
               
    return (t1/t2)

def probOptUABNUM(u, a2, b2, mu, mu2):
    n, d = probOptUAB(u, a2, b2)
    t1 = matNum(n, mu, mu2)
    t2 = matNum(d, mu, mu2)
            
    return (t1/t2)

def KBase(s, f, s2, f2, ss, ff, ss2, ff2):
    if s == 0 and f == 0 and s2 == 0 and f2 == 0:
        return 1
    elif s == -1 or f == -1 or s2 == -1 or f2 == -1 :
        return 0
    else:
        return (KBase(s - 1, f, s2, f2, ss, ff, ss2, ff2) * probOpt(ss + s - 1, ff + f, ss2 + s2, ff2 + f2)
        +  KBase(s, f - 1, s2, f2, ss, ff, ss2, ff2) * probOpt(ss + s, ff + f - 1, ss2 + s2, ff2 + f2)
        +  KBase(s, f, s2 - 1, f2, ss, ff, ss2, ff2) * (1 - probOpt(ss + s, ff + f, ss2 + s2 - 1, ff2 + f2))
        +  KBase(s, f, s2, f2 - 1, ss, ff, ss2, ff2) * (1 - probOpt(ss + s, ff + f, ss2 + s2, ff2 + f2 - 1)))
    

def probOptBase(s, f, u2, ss, ff, ss2, ff2):
    n = 0
    d = 0
    for s2 in range(u2 + 1):
        f2 = u2 - s2
        n = n + KBase(s, f, s2, f2, ss, ff, ss2, ff2) * probMeanState (s, f, s2, f2) * probOpt(s + ss, f + ff, s2 + ss2, f2 + ff2)
        d = d + KBase(s, f, s2, f2, ss, ff, ss2, ff2) * probMeanState (s, f, s2, f2)

    return n, d


def probOptBaseNum(mu, mu2, s, f, u2, ss, ff, ss2, ff2):
    n = 0
    d = 0
    for s2 in range(u2 + 1):
        f2 = u2 - s2
        n = n + matNum(KBase(s, f, s2, f2, ss, ff, ss2, ff2) * probMeanState (s, f, s2, f2) * probOpt(s + ss, f + ff, s2 + ss2, f2 + ff2), mu, mu2)
        #print(n)
        d = d + matNum(KBase(s, f, s2, f2, ss, ff, ss2, ff2) * probMeanState (s, f, s2, f2), mu, mu2)
        #print(d)
    return n/d

def probOptBaseT(mu, mu2, t, ss, ff, ss2, ff2):
    n = 0
    d = 0
    for s in range(t + 1):
        for f in range(t - s + 1):
            for s2 in range(t - s - f + 1):
            
                f2 = t - s - f - s2
                n = n + matNum(KBase(s, f, s2, f2, ss, ff, ss2, ff2) * probMeanState (s, f, s2, f2) * probOpt(s + ss, f + ff, s2 + ss2, f2 + ff2), mu, mu2)
                #print(n)
                d = d + matNum(KBase(s, f, s2, f2, ss, ff, ss2, ff2) * probMeanState (s, f, s2, f2), mu, mu2)
                #print(d)
    return n/d


    
#k = proKState(s, f, s2, f2)
#np.set_printoptions(formatter={'all':lambda x: str(Fraction(x).limit_denominator(100000))})

    

mu = 0.8
mu2 = 0.6
s = 1
f = 1
u2 = 1
ss = 0
ff = 0
ss2 = 0
ff2 = 0
T = 4

t = 2
#probOpt(ss,ff,ss2,ff2)*mu*probOptBaseNum(mu,mu2,s-1,f,u2,ss+1,ff,ss2,ff2) + probOpt(ss,ff,ss2,ff2)*(1-mu)*probOptBaseNum(mu,mu2,s,f-1,u2,ss,ff+1,ss2,ff2) + (1-probOpt(ss,ff,ss2,ff2))*mu2*probOptBaseNum(mu,mu2,s,f,u2-1,ss,ff,ss2+1,ff2) + (1-probOpt(ss,ff,ss2,ff2))*(1-mu2)*probOptBaseNum(mu,mu2,s,f,u2-1,ss,ff,ss2,ff2+1)
##a = np.zeros(T+1)
##b = np.zeros(T+1)
##c = np.zeros(T+1)

##for i in range(T + 1):
##	a[i]=probOptABNUM(i,T-i,Fraction(9,10),Fraction(8,10))
##	b[i]=probOptABNUM(i,T-i,Fraction(6,10),Fraction(4,10))
##	c[i]=probOptABNUM(i,T-i,Fraction(2,10),Fraction(1,10))
##
##plt.plot(a, label = '0.9 - 0.8')
##plt.plot(b, label = '0.6 - 0.4')
##plt.plot(c, label = '0.2 - 0.1')
##plt.legend(loc = 'upper left')
##plt.savefig('UU2_' + str(T) + '.png')
##plt.show()
##    
##for i in range(T + 1):
##	a[i]=probOptSFNUM(i,T-i,Fraction(9,10),Fraction(8,10))
##	b[i]=probOptSFNUM(i,T-i,Fraction(6,10),Fraction(4,10))
##	c[i]=probOptSFNUM(i,T-i,Fraction(2,10),Fraction(1,10))
##
##plt.plot(a, label = '0.9 - 0.8')
##plt.plot(b, label = '0.6 - 0.4')
##plt.plot(c, label = '0.2 - 0.1')
##plt.legend(loc = 'upper left')
##plt.savefig('SF_' + str(T) + '.png')
##plt.show()

L = np.arange(1000)/1000
for i in range(11, 11+1):
    temp = probPOptN(i) - probOptN(i)
    print(i)
    for mu in L:
        for mu2 in L:
            if mu > mu2:
                flg = matNum(temp, mu, mu2)
                if flg < 0:
                    print(i, mu, mu2, flg)

    #print(matNum(temp, 0.999999, 0.99), matNum(temp, 0.99, 0.79), matNum(temp, 0.6, 0.4))

#recursively multiply probabilities and arrays to get matrices with probabilities
