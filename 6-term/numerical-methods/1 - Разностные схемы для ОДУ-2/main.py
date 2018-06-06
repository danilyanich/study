import numpy as np
import math as ma
import matplotlib.pyplot as plt

f = lambda x: 1 / ma.cos(x)**2
a, b = 0, 0.5
n = 10

h = (b - a) / n
x = [a + (i * h) for i in range(0, n + 1)]


def main():
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

  return A, B



if __name__ == "__main__":
  A, B =
  y = np.linalg.solve(A, B)

  plt.plot(x, y)
  plt.show()

  print(y, x)

  for a in A:
    for e in a:
      print("{:10.2f}".format(e), end="")
    print()
