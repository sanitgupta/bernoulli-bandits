import numpy as np
import math
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint

epsilon = 0.1
delta = 0.1

k1 = 5/4

n = 50
#qt = np.append(0.6, 0.49 * np.ones(n - 1))
qt = np.arange(0, n) * 1 / n + 1 / (2 * n)
#qt = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
print(qt)
n = len(qt)

t = 100

end_times = np.zeros(t)
correctness = np.zeros(t)
perfection = np.zeros(t)

for i in range(t):
    f = np.zeros(n)
    s = np.zeros(n)
    U = np.zeros(n)
    for j in range(0, n):
        d = j

        if qt[d] > np.random.random():
            r = 1
            s[d] = s[d] + 1
        else : 
            r = 0
            f[d] = f[d] + 1

        U[d] = U[d] + 1 

    while True:
        mean = s/U
        bnd = np.sqrt(np.log((k1 * n) / (delta))/(2*U)) #np.sqrt(np.log((k1 * n * np.sum(U) ** 4) / (delta))/(2*U))
        upper = mean + bnd
        lower = mean - bnd

        upper = np.ma.array(upper, mask = False)
        
        d1 = np.argmax(mean)

        upper.mask[d1] = True
        
        d2 = np.argmax(upper)

        upper.mask[d2] = False

        if upper[d2] - lower[d1] < epsilon: #stopping condition
            break

        if qt[d1] > np.random.random():
            r = 1
            s[d1] = s[d1] + 1
        else : 
            r = 0
            f[d1] = f[d1] + 1

        U[d1] = U[d1] + 1

        if qt[d2] > np.random.random():
            r = 1
            s[d2] = s[d2] + 1
        else : 
            r = 0
            f[d2] = f[d2] + 1

        U[d2] = U[d2] + 1 
        
    end_times[i] = np.sum(U)

    if qt[d1] >= max(qt) - epsilon:
        correctness[i] = 1
        
    if qt[d1] == max(qt):
        perfection[i] = 1
    
    print("Iteration " + str(i) + ": Terminated in " + str(np.sum(U)) + " time steps")

print("For epsilon = " + str(epsilon) + ", delta = " + str(delta) + " and bandit_instance : " + str(qt))     
print(str(np.mean(end_times)) + " +- " + str(np.std(end_times)/np.sqrt(t)))
print("Correctness fraction = " + str(np.sum(correctness)/t))
print("Optimal arm identfied fraction = " + str(np.sum(perfection)/t))
