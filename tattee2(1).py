import numpy as np
import math
from math import factorial as fac
from fractions import Fraction

##Fraction('3.1415926535897932').limit_denominator(1000

#Prob(choosing optimal arm)
def probOpt(s, f, s2, f2):
    if s < 0 or f < 0 or s2 < 0 or f2 < 0:
        return 0
    res = 0
    for i in range(f + 1):
        for j in range(f2 + 1):
            res = res + Fraction((-1)**(i+j), fac(f-i)*fac(f2-j)*fac(i)*fac(j)*(s2+j+1)*(s+s2+i+j+2))

    return Fraction(res * fac(s+f+1) * fac(s2+f2+1), fac(s) * fac(s2))

def ex(x, n):
    t = 1
    for i in range(n):
        t = t*x
    return t
    
#mean coefficient of prob(existing in that state)
def probMeanState(s, f, s2, f2):
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


def multivariate(sum,a,b,c,d):
	return Fraction(fac(sum), fac(a)*fac(b)*fac(c)*fac(d))



def ans(n, bat):
    tatty = Fraction(0)
    for a in range(n + 1):
        for b in range(n + 1):
            for c in range(n + 1):
                d = n - a - b - c
                if d < 0:
                    continue
                if (a + b + c + d) == 0:
                        continue
                if (a + b + c + d) % bat == 0:
                        prev = a + b + c + d
                else:
                        prev = a + b + c + d - (a+b+c+d)%bat

                for e in range(a + 1):
                    for f in range(b + 1):
                        for g in range(c + 1):
                            h = prev - e - f - g
                            if h < 0 or h > d:
                                continue
                            
                            tatty += tatti[e][f][g][h] * ex(probOpt(e, f, g, h),a + b - e - f) * ex(1 - probOpt(e, f, g, h),c + d - g - h) * multivariate(a+b+c+d-e-f-g-h,a-e,b-f,c-g,d-h) * probOpt(e,f,g,h) * probMeanState(a,b,c,d)

    return tatty

##np.set_printoptions(formatter={'all':lambda x: str(Fraction(x).limit_denominator(100000))})

T = 4

   
tatti = np.zeros((11,11,11,11))
tatti = tatti.astype('int') + Fraction()
tatti[0][0][0][0] = Fraction(1)


L = 10
batch = 1
for l in range(1,L+1):
	for a in range(l + 1):
		for b in range(l + 1):
			for c in range(l + 1):
				for d in range(l + 1):
					if(a + b + c + d == l):
						
						if (a + b + c + d) == 0:
							continue
						if (a + b + c + d) % batch == 0:
							prev = a + b + c + d - batch
						else:
							prev = a + b + c + d - (a+b+c+d)%batch

						for e in range(a + 1):
							for f in range(b + 1):
								for g in range(c + 1):
									for h in range(d + 1):
										if e + f + g + h == prev:
										    tatti[a][b][c][d] += tatti[e][f][g][h] * ex(probOpt(e, f, g, h), a + b - e - f) * ex(1 - probOpt(e, f, g, h),c + d - g - h) * multivariate(a+b+c+d-e-f-g-h,a-e,b-f,c-g,d-h)
b1= ans(T, 1)


tatti = np.zeros((11,11,11,11))
tatti = tatti.astype('int') + Fraction()
tatti[0][0][0][0] = Fraction(1)

L = 10
batch = 2
for l in range(1,L+1):
	for a in range(l + 1):
		for b in range(l + 1):
			for c in range(l + 1):
				for d in range(l + 1):
					if(a + b + c + d == l):
						
						if (a + b + c + d) == 0:
							continue
						if (a + b + c + d) % batch == 0:
							prev = a + b + c + d - batch
						else:
							prev = a + b + c + d - (a+b+c+d)%batch

						for e in range(a + 1):
							for f in range(b + 1):
								for g in range(c + 1):
									for h in range(d + 1):
										if e + f + g + h == prev:
											tatti[a][b][c][d] += tatti[e][f][g][h] * (probOpt(e, f, g, h)**(a + b - e - f)) * ((1 - probOpt(e, f, g, h))**(c + d - g - h)) * multivariate(a+b+c+d-e-f-g-h,a-e,b-f,c-g,d-h)
														

b2 = ans(T,2)														

c = b1 - b2
print(b1 - b2)
mu = Fraction(999,1000)
mu2 = Fraction(1,1000)


t = Fraction(0)

for i in range(len(c)):
    for j in range(len(c)):
	    t += c[i][j] * ex(mu,i) * ex(mu2,j)

print(t)

y= np.zeros(10)
z= np.zeros(10)
for i in range(10 + 1):
	y[i]=probOptABNUM(i,10-i,Fraction(6,10),Fraction(4,10))
	z[i]=probOptABNUM(i,10-i,Fraction(4,10),Fraction(4,10))

plt.plot(y, label = '0.6 - 0.4')
plt.plot(y, label = '0.4 - 0.4')
plt.show()
