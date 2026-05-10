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
    y_predict = old_a * X + old_b

    error = Y - y_predict

    derivative_a = -2 * np.mean(X * error)
    derivative_b = -2 * np.mean(error)

    new_a = old_a - learning_rate * derivative_a
    new_b = old_b - learning_rate * derivative_b
    return new_a, new_b

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

X_data = line.P[:, 0]
Y_data = line.P[:, 1]

for i in range(epochs):
    A,B = gradient_descent(A, B, LR, X_data, Y_data)
    if i % 50 == 0:
        current_loss = loss(A, B, X_data, Y_data)
        print(f"Epoch {i:3d} | Loss: {current_loss:.4f} | a: {A:.3f}, b: {B:.3f}")

print("-" * 30)
print(f"Trọng số TỐI ƯU cuối cùng: A = {A:.3f}, B = {B:.3f}")
# (Đáng lẽ phải tiến gần tới a = -2, b = 4 theo công thức sinh dữ liệu của bạn)

plt.scatter(X_data, Y_data, c = line.L, s = 40)

y_learned = A * line.P[:, 0] + B
plt.plot(line.P[:, 0], y_learned, color='red', linewidth=3, label=f"Đường hồi quy: y = {A:.2f}x + {B:.2f}")

plt.title("Mô phỏng Gradient Descent cho Hồi quy tuyến tính")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()