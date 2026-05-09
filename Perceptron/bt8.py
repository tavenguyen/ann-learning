import numpy as np
import matplotlib.pyplot as plt
import Dense as dense
import Loss as loss

class Spiral:
    def __init__(self, n_points, n_classes, n_dimensions):
        self.N = n_points
        self.D = n_dimensions
        self.K = n_classes

        self.P = np.zeros((self.N * self.K, self.D))
        self.L = np.zeros(self.N * self.K, dtype = 'uint8')
        for j in range(self.K):
            r_max = 2 * (j + 1)
            t = np.linspace(0, 1, self.N)
            r = r_max * t
            theta = 2 * 2 * np.pi * t + (j * 2 * np.pi) / self.K + np.random.randn(self.N) * 0.5
            
            ix = range(self.N * j, self.N * (j + 1))
            if self.D == 2:
                self.P[ix] = np.c_[r * np.cos(theta), r * np.sin(theta)]
            self.L[ix] = j

# Initial Parameters
N = 1000
K = 3
D = 2

inputs = Spiral(N, K, D)

plt.scatter(inputs.P[:, 0], inputs.P[:, 1], c=inputs.L, s=40, cmap=plt.cm.Spectral)
plt.show()

layer1 = dense.DenseLayer(n_inputs=2, n_neurons=3)
layer2 = dense.DenseLayer(n_inputs=3, n_neurons=3)
layer1_output = layer1.forward(inputs.P, activation = dense.relu)
layer2_output = layer2.forward(layer1_output, activation=dense.softmax)

loss_function = loss.Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(layer2_output, inputs.L)

predictions = np.argmax(layer2_output, axis = 1)
print(f"Loss: {loss:.2f}")

plt.scatter(inputs.P[:, 0], inputs.P[:, 1], c=predictions, s=40, cmap=plt.cm.Spectral)
plt.show()