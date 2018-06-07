import numpy as np
import scipy as sp
import scipy.integrate
import math as ma
import matplotlib.pyplot as plt

q_x = lambda x: x
k_x = lambda x: 2 - x
_1_k_x = lambda x: 1 / k_x(x)
f_x = lambda x: 2*ma.cos(x) - ma.sin(x)

_a, _b = 0, 1

def integrate(func, _from, to):
  return sp.integrate.quad(func, _from, to)[0]


def balance(n):
  h = (_b - _a) / n
  x = [_a + (i * h) for i in range(0, n + 1)]

  d = np.zeros(n + 1)
  d[0] = 2/h * integrate(q_x, 0, h/2)
  d[n] = 2/h * integrate(q_x, 1 - h/2, 1)

  for i in range(1, n):
    d[i] = 1/h * integrate(q_x, x[i] - h/2, x[i] + h/2)


  a = np.zeros(n + 1)

  for i in range(1, n + 1):
    a[i] = 1 / (1/h * integrate(_1_k_x, x[i - 1], x[i]))


  phi = np.zeros(n + 1)
  phi[0] = 2/h * integrate(f_x, 0, h/2)
  phi[n] = 2/h * integrate(f_x, 1 - h/2, 1)

  for i in range(1, n):
    phi[i] = 1/h * integrate(f_x, x[i] - h/2, x[i] + h/2)


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

  return np.linalg.solve(A, B), x


def ritzh(n):
  h = (_b - _a) / n
  x = [_a + (i * h) for i in range(0, n + 1)]

  # d = np.zeros(n + 1)

  # for i in range(1, n- 1)
  #   d[i] = 1/h * sp.integrate.quad(lambda x: q_x(x)*(x - x[i - i]), x[i - 1], x[i])[0] -
  #     sp.integrate.quad(lambda x: )


  # d0= 2/h**2 * sp.integrate.quad(lambda x: q_x(x)*(h - x), 0, h)[0]
  # dN= 2/h**2 * sp.integrate.quad(lambda x: q_x(x)*(x - 1 + h), 1 - h, 1)[0]
  # f0= 2/h**2 * sp.integrate.quad(lambda x: f_x(x)*(h - x), 0, h)[0]
  # fN= 2/h**2 * sp.integrate.quad(lambda x: f_x(x)*(x - 1 + h), 1 - h, 1)[0]


  A = np.identity(n + 1)

  A[0, 0] = 1/h**2 * (integrate(k_x, 0, h) + integrate(lambda _x: q_x(_x) * (_x - h)**2, 0, h)) +  1
  A[n, n] = 1/h**2 * (integrate(k_x, 1 - h, 1) + integrate(lambda _x: q_x(_x) * (_x - 1 + h)**2, 1 - h, 1)) + ma.tan(1)

  for i in range(1, n):
    res = 1/h**2 * (integrate(lambda _x: q_x(_x) * (x[i + 1] - _x) * (_x - x[i]), x[i], x[i + 1]) - integrate(k_x, x[i], x[i + 1]))
    A[i, i + 1] = res
    A[i + 1, i] = res

  for i in range(1, n):
    A[i, i] = 1/h**2 * (integrate(k_x, x[i - 1], x[i + 1]) + integrate(lambda _x: q_x(_x)*(_x - x[i - 1])**2, x[i - 1], x[i]) + integrate(lambda _x: q_x(_x)*(x[i + 1] - _x)**2, x[i], x[i + 1]))


  # def Ai(xi_1, xi):
  #   return 1/h * (sp.integrate.quad(lambda x: k_x(x), xi_1, xi)[0] - sp.integrate.quad(lambda x: q_x(x)*(x - xi_1)*(xi - x), xi_1, xi)[0])

  # def Di(xi_1,xi,xi_2):
  #   return 1/h**2 * (sp.integrate.quad(lambda x: q_x(x)*(x - xi_1), xi_1, xi)[0] - sp.integrate.quad(lambda x: q_x(x)*(xi - x), xi, xi_2)[0])

  # A[0, 0] = -Ai(0, h)/h - (1 + h/2 * d0)
  # A[0, 1] = Ai(0, h)/h

  # A[n, n - 1] = Ai(1 - h, 1)/h
  # A[n, n] = -Ai(1 - h, 1)/h - (ma.tan(1) + h/2 * dN)

  # for i in range(1,n):
  #   A[i, i + 1] = Ai(x[i], x[i+1])/h**2
  #   A[i, i] = -((Ai(x[i], x[i+1]) + Ai(x[i-1], x[i]))/h**2 + Di(x[i - 1], x[i], x[i + 1]))
  #   A[i,i - 1] = Ai(x[i - 1], x[i])/h**2


  B = np.zeros(n + 1)

  B[0] = 1/h * integrate(lambda _x: f_x(_x) * (_x - h), 0, h) + 1
  B[n] = 1/h * integrate(lambda _x: f_x(_x) * (_x - 1 + h), 1 - h, 1)

  for i in range(1, n):
    B[i] = 1/h * (integrate(lambda _x: f_x(_x) * (_x - x[i - 1]), x[i - 1], x[i]) + integrate(lambda _x: f_x(_x) * (x[i+1] - _x), x[i], x[i + 1]))


  # def fi(xi_1, xi, xi_2):
  #   return 1/h**2 * (sp.integrate.quad(lambda x: f_x(x)*(x - xi_1), xi_1, xi)[0] - sp.integrate.quad(lambda x: f_x(x)*(xi-x), xi, xi_2)[0])

  # B[0] = -(1 + h/2*f0)
  # B[n] = -h/2*fN

  # for i in range(1,n):
  #   B[i] = -fi(x[i - 1], x[i], x[i + 1])

  for r in A:
    for e in r:
      print('{:15.5}'.format(e), end='')
    print()

  print(B)

  return np.linalg.solve(A, B), x


if __name__ == "__main__":

  for n in range(1, 2):
    solutionBalance, x = balance(10 * n)
    plt.plot(x, solutionBalance)

  for n in range(1, 10):
    solutionRitzh, x = ritzh(10 * n)
    plt.plot(x, solutionRitzh)

  plt.show()
