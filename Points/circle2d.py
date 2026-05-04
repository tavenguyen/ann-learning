import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2
K = 3

P = np.zeros((N * K, D))
L = np.zeros(N * K, dtype = 'uint8')

for j in range(K):
    # 2, 4, 6
    r = 2 * (j + 1)
    r_noise = r + np.random.randn(N) * 2
    
    theta = np.linspace(0, 2 * np.pi, N)
    idx = range(N * j, N * (j + 1))
    if D == 2:
        P[idx] = np.c_[r_noise * np.cos(theta), r_noise * np.sin(theta)]
    L[idx] = j

plt.scatter(P[:, 0], P[:, 1], c = L, s = 40)
plt.title("Hình tròn")
plt.xlabel('Trục x')
plt.ylabel('Trục y')
plt.show()
