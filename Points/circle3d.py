import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 3
K = 3

P = np.zeros((N * K, D))
L = np.zeros(N * K, dtype = 'uint8')

for j in range(K):
    radius = 2 * (j + 1) + np.random.randn(N) * (j + 1)
    azimuth = np.linspace(0, 2 * np.pi, N)
    polar = np.linspace(0, np.pi, N)

    idx = range(N * j, N * (j + 1))
    if D == 3:
        P[idx] = np.c_[radius * np.cos(polar) * np.cos(azimuth), radius * np.cos(polar) * np.sin(azimuth), radius * np.sin(azimuth)]
    L[idx] = j

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Vẽ các điểm dữ liệu
scatter = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=L, s=20, cmap='Spectral', alpha=0.6)

ax.set_title(f"Dataset {K} Lớp Hình Cầu Đồng Tâm", fontsize=14)
ax.set_xlabel('X (Feature 1)')
ax.set_ylabel('Y (Feature 2)')
ax.set_zlabel('Z (Feature 3)')

# Thêm thanh chú thích màu sắc
plt.colorbar(scatter, ax=ax, label='Class Label')
plt.show()