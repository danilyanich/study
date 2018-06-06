import numpy as np
import math as ma

f = lambda x: 1 / ma.cos(x)**2
a, b = 0, 0.5
n = 10

h = (b - a) / n
x = [a + (i * h) for i in range(0, n + 1)]

A = np.identity(n + 1)
