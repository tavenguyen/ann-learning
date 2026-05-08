import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2
K = 1

P = np.zeros((N * K, D))
L = np.zeros(N * K, dtype = 'uint8')

for j in range(K):
    t = np.linspace(0, 1, N)
    theta = t * 2 * np.pi + 2 * j * np.pi * 1/3
    r_noise = t

    idx = range(N * j, N * (j + 1))
    if D == 2:
        P[idx] = np.c_[r_noise * np.cos(theta), r_noise * np.sin(theta)]

    L[idx] = j

plt.scatter(P[:, 0], P[:, 1], c = L, s = 40)
plt.show()