import numpy as np
import matplotlib.pyplot as plt

class Line: 
    def __init__(self, n_inputs: int, n_classes: int, n_dimensions: int):
        self.N = n_inputs
        self.K = n_classes
        self.D = n_dimensions

        self.P = np.zeros((self.N * self.K, self.D))
        self.L = np.zeros(self.N * self.K)

        for j in range(self.K):
            a = 2 * (j - 1)
            b = 2 * (j + 2) + np.random.randn(self.N) * 2

            ix = range(self.N * j, self.N * (j + 1))
            t = np.linspace(-10, 10, self.N)
            if self.D == 2:
                self.P[ix] = np.c_[t, a * t + b]
            self.L[ix] = j

# inputs(x,y)
def gradient_descent(old_a, old_b, learning_rate, X, Y):
    derivative_x = 0
    derivative_y = 0
    for j in range(len(X)):
        derivative_x += -(2/len(X)) * X[j] * (Y[j] - (old_a * X[j] + old_b))
        derivative_y += -(2/len(X)) * (Y[j] - (old_a * X[j] + old_b))

    A = old_a - learning_rate * derivative_x
    B = old_b - learning_rate * derivative_y
    return A, B

def loss(a, b, x, y):
    y_predict = a * x + b
    sample_losses = (y - y_predict) ** 2
    loss = np.mean(sample_losses)
    return loss

N = 100
K = 1
D = 2

line = Line(N, K, D)

A = 0
B = 0
LR = 0.01
epochs = 5000

for i in range(epochs):
    A,B = gradient_descent(A, B, LR, line.P[:, 0], line.P[:, 1])
    print(loss(A, B, line.P[:, 0], line.P[:, 1]))

print(A, B)

plt.scatter(line.P[:, 0], line.P[:, 1], c = line.L, s = 40)

y_learned = A * line.P[:, 0] + B
plt.plot(line.P[:, 0], y_learned, color='red', linewidth=3, label=f"Đường hồi quy: y = {A:.2f}x + {B:.2f}")

plt.show()

