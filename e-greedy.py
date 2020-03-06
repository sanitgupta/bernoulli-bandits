import numpy as np
import matplotlib.pyplot as plt
from random import randint

def choice():
    p = np.random.random()
    if p < e :
        return randint(0, n - 1)
    else:
        return np.argmax(q)

n = 20
T = 1000
t = 1000
e = 0.01

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

plt.plot(reg, label = e)

plt.legend(loc = 'upper left')
plt.show()
