import numpy as np
import math as ma
import matplotlib.pyplot as plt

f = lambda x: 1 / ma.cos(x)**2
_a, _b = 0, 0.5


def solve(n):
  h = (_b - _a) / n
  x = [_a + (i * h) for i in range(0, n + 1)]

  A = np.identity(n + 1)
  A[0, 0] = -1 / h
  A[0, 1] = 1 / h

  A[n, n - 1] = 0
  A[n, n] = 1

  for i in range(1, n):
    A[i, i - 1] = 1 / h**2
    A[i, i] = -(2 / h**2 + 1 / h + 2 / ma.cos(x[i]) ** 2)
    A[i, i + 1] = 1 / h**2 + 1 / h

  B = np.zeros(n + 1)
  B[0] = ma.cos(0.5) ** 2

  for i in range(1, n - 1):
    B[i] = 1 / ma.cos(x[i]) ** 2

  return np.linalg.solve(A, B), x


def solve_2(n):
  h = (_b - _a) / n
  x = [_a + (i * h) for i in range(0, n + 1)]

  al_0 = -2*h / (2 - h) + 1
  be_0 = 2 / (2 - h)
  ga_0 = h / (2 - h)

  al_1 = h**2 / ma.cos(0.5)**2
  be_1 = -h
  ga_1 = -ma.cos(0.5)**2 + h**2 / (2*ma.cos(0.5)**2)

  A = np.identity(n + 1)
  A[0, 0] = al_0 - 1/h * be_0
  A[0, 1] = 1/h * be_0

  A[n, n - 1] = -1/h * be_1
  A[n, n] = al_1 + 1/h * be_1

  for i in range(1, n):
    A[i, i - 1] = 1 / h**2
    A[i, i] = -(2 / h**2 + 1 / h + 2 / ma.cos(x[i]) ** 2)
    A[i, i + 1] = 1 / h**2 + 1 / h

  B = np.zeros(n + 1)
  B[0] = ga_0
  B[n] = ga_1

  for i in range(1, n):
    B[i] = 1 / ma.cos(x[i]) ** 2

  return np.linalg.solve(A, B), x



if __name__ == "__main__":
  for n in range(1, 10):
    solution, x = solve(10 * n)
    plt.plot(x, solution)


  for n in range(999, 1000):
    solution_2, x = solve_2(10 * n)
    plt.plot(x, solution_2)

  plt.show()
