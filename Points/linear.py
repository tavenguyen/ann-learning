import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of points per class
D = 2 # Dimensions
K = 3 # number of classes

P = np.zeros((N * K, D))
L = np.zeros(N * K, dtype = 'uint8')
# y = ax + b
for j in range(K):
    a = 2 * (j - 1)
    b = (j - 2) * 2 + np.random.randn(N) * 2  
    ix = range(N * j, N * (j + 1))
    t = np.linspace(-10, 10, N)
    if D == 2:
        P[ix] = np.c_[t, a * t + b]

    L[ix] = j

plt.scatter(P[:, 0], P[:, 1], c=L, s=40, cmap=plt.cm.Spectral)
plt.title(f"Tạo ngẫu nhiên {K} nhóm điểm dạng đường thẳng")
plt.xlabel("Trục X")
plt.ylabel("Trục Y")
plt.show()