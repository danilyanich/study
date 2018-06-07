import numpy as np
import scipy as sp
import scipy.integrate
import math as ma
import matplotlib.pyplot as plt

q_x = lambda x: x
_1_k_x = lambda x: 1 / (2 - x)
f_x = lambda x: 2*ma.cos(x) - ma.sin(x)

a, b = 0, 1
n = 10

h = (b - a) / n
x = [a + (i * h) for i in range(0, n + 1)]

def balance():
  d = np.zeros(n + 1)
  d[0] = 2/h * sp.integrate.quad(q_x, 0, h/2)[0]
  d[n] = 2/h * sp.integrate.quad(q_x, 1 - h/2, 1)[0]

  for i in range(1, n):
    d[i] = 1/h * sp.integrate.quad(q_x, x[i] - h/2, x[i] + h/2)[0]


  a = np.zeros(n + 1)

  for i in range(1, n + 1):
    a[i] = 1 / (1/h * sp.integrate.quad(_1_k_x, x[i - 1], x[i])[0])


  phi = np.zeros(n + 1)
  phi[0] = 2/h * sp.integrate.quad(f_x, 0, h/2)[0]
  phi[n] = 2/h * sp.integrate.quad(f_x, 1 - h/2, 1)[0]

  for i in range(1, n):
    phi[i] = 1/h * sp.integrate.quad(f_x, x[i] - h/2, x[i] + h/2)[0]


  A = np.identity(n + 1)
  A[0, 0] = a[1]/h - 1 + h/2 * d[0]
  A[0, 1] = a[1]/h

  A[n, n - 1] = a[n]/h
  A[n, n] = -a[n]/h + ma.tan(1) + h/2*d[n]

  for i in range(1, n):
    A[i, i - 1] = a[i] / h**2
    A[i, i] = -((a[i + 1] + a[i]) / h**2 + d[i])
    A[i, i + 1] = a[i + 1] / h**2


  B = np.zeros(n + 1)
  B[0] = -1 - h/2 * phi[0]
  B[n] = h/2 * phi[n]

  for i in range(1, n):
    B[i] = - phi[i]

  return np.linalg.solve(A, B)


if __name__ == "__main__":
  solutionBalance = balance()

  print('Баланс:')
  print(solutionBalance)

  plt.plot(x, solutionBalance)
  plt.show()
