import numpy as np
import math
from math import factorial as fac
from fractions import Fraction

##Fraction('3.1415926535897932').limit_denominator(1000


def ex(x, n):
    t = 1
    for i in range(n):
        t = t*x
    return t

def probOpt(s, f, s2, f2):
    if s < 0 or f < 0 or s2 < 0 or f2 < 0:
        return 0
    res = 0
    for i in range(f + 1):
        for j in range(f2 + 1):
            res = res + Fraction((-1)**(i+j), fac(f-i)*fac(f2-j)*fac(i)*fac(j)*(s2+j+1)*(s+s2+i+j+2))

    return Fraction(res * fac(s+f+1) * fac(s2+f2+1), fac(s) * fac(s2))


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

def probState(s, f, s2, f2):
    if s + f + s2 + f2 == 0:
        return 1
    return K(s, f, s2, f2) * probMeanState(s, f, s2, f2)

def probStateNUM(mu, mu2, s, f, s2, f2):
    x = probState(s,f,s2,f2)

    t = 0
    for i in range(len(x)):
        for j in range(len(x)):
            t = t + ex(mu,i) * ex(mu2,j) * x[i,j]

    return t
            

def probOptN(n):
    z = 0
    for s in range(n + 1):
        for f in range(n + 1):
            for s2 in range(n + 1):
                for f2 in range(n + 1):
                    if s + f + s2 + f2 == n:
                        z = z + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)

    return z



def matNum(x, mu, mu2):
    t = 0
    for i in range(len(x)):
        for j in range(len(x)):
            t = t + ex(mu,i) * ex(mu2,j) * x[i,j]

    return t





##


def probOptNext(s, f, s2, f2):
    return probOpt(s, f, s2, f2) * (probMeanState(1, 0, 0, 0) * probOpt(s + 1, f, s2, f2) + probMeanState(0, 1, 0, 0) * probOpt(s, f + 1, s2, f2)) + (1 - probOpt(s, f, s2, f2)) * (probMeanState(0, 0, 1, 0) * probOpt(s, f, s2 + 1, f2) + probMeanState(0, 0, 0, 1) * probOpt(s, f, s2, f2 + 1))

def probOptAB(a, b):
    n = 0
    d = 0
    for s in range(a + 1):
        for f in range(a + 1):
            if s + f == a:
                for s2 in range(b + 1):
                    for f2 in range(b + 1):
                        if s2 + f2 == b:
                            n = n + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)
                            d = d + probState(s, f, s2, f2) 
    return n, d

def probOptABU2(s, f, b):
    n = 0
    d = 0

    for s2 in range(b + 1):
        for f2 in range(b + 1):
            if s2 + f2 == b:
                n = n + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)
                d = d + probState(s, f, s2, f2)
    return n, d

def probOptUAB(a, s2, f2):
    n = 0
    d = 0

    for s in range(a + 1):
        for f in range(a + 1):
            if s + f == a:
                n = n + probOpt(s, f, s2, f2) * probState(s, f, s2, f2)
                d = d + probState(s, f, s2, f2)
    return n, d
#def proState(s, f, s2, f2):
def probOptABNUM(a, b, mu, mu2):
    n, d = probOptAB(a, b)
    t1 = 0
    t2 = 0
    for i in range(len(n)):
        for j in range(len(n)):
            t1 = t1 + n[i][j] * ex(mu,i) * ex(mu2, j)

    for i in range(len(d)):
        for j in range(len(d)):
            t2 = t2 + d[i][j] * ex(mu,i) * ex(mu2, j)
            
    return float(t1/t2)

def probOptABU2NUM(a, b, u2, mu, mu2):
    n, d = probOptABU2(a, b, u2)
    t1 = 0
    t2 = 0
    for i in range(len(n)):
        for j in range(len(n)):
            t1 = t1 + n[i][j] * ex(mu,i) * ex(mu2, j)

    for i in range(len(d)):
        for j in range(len(d)):
            t2 = t2 + d[i][j] * ex(mu,i) * ex(mu2, j)
            
    return (t1/t2)

def probOptUABNUM(u, a2, b2, mu, mu2):
    n, d = probOptUAB(u, a2, b2)
    t1 = 0
    t2 = 0
    for i in range(len(n)):
        for j in range(len(n)):
            t1 = t1 + n[i][j] * ex(mu,i) * ex(mu2, j)

    for i in range(len(d)):
        for j in range(len(d)):
            t2 = t2 + d[i][j] * ex(mu,i) * ex(mu2, j)
            
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

#probOpt(ss,ff,ss2,ff2)*mu*probOptBaseNum(mu,mu2,s-1,f,u2,ss+1,ff,ss2,ff2) + probOpt(ss,ff,ss2,ff2)*(1-mu)*probOptBaseNum(mu,mu2,s,f-1,u2,ss,ff+1,ss2,ff2) + (1-probOpt(ss,ff,ss2,ff2))*mu2*probOptBaseNum(mu,mu2,s,f,u2-1,ss,ff,ss2+1,ff2) + (1-probOpt(ss,ff,ss2,ff2))*(1-mu2)*probOptBaseNum(mu,mu2,s,f,u2-1,ss,ff,ss2,ff2+1)
##y= np.zeros(10)
##z= np.zeros(10)
##for i in range(10 + 1):
##	y[i]=probOptABNUM(i,10-i,Fraction(6,10),Fraction(4,10))
##	z[i]=probOptABNUM(i,10-i,Fraction(4,10),Fraction(4,10))
##
##plt.plot(y, label = '0.6 - 0.4')
##plt.plot(z, label = '0.4 - 0.4')
##plt.show()
    

##s = 0
##f = 1
##s2 = 2
##f2 = 0
##print(fac(5))
##
##p = dblquad(pro, 0, 1, lambda x: 0, lambda x: x, args = (s, f, s2, f2))[0]
##
###for i in range(0, n
##
##print(Fraction(p).limit_denominator(1000))


#recursively multoply probabilities and arrays to get matrices with probabilities
