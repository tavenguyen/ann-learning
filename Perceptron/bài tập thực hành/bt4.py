import numpy as np
import matplotlib.pyplot as plt

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

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.max(x, 0)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs, activation = None):
        self.output = np.dot(inputs, self.weights) + self.biases

        # activation function
        if activation is not None:
            self.output = activation(self.output)

        return self.output
    
input = Spiral(n_points=1000, n_classes=3, n_dimensions=2)
plt.scatter(input.P[:, 0], input.P[:, 1], c = input.L, s = 40, cmap=plt.cm.Spectral)
plt.show()

layer1 = DenseLayer(n_inputs=2, n_neurons=3)
layer2 = DenseLayer(n_inputs=3, n_neurons=3)
layer1.forward(input.P, activation=relu)
layer2.forward(layer1.output, activation=softmax)
predictions = np.argmax(layer2.output,axis=1)

plt.scatter(input.P[:, 0], input.P[:, 1], c = predictions, s = 40, cmap=plt.cm.Spectral)
plt.show()