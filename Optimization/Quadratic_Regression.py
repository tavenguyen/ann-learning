import numpy as np
import matplotlib.pyplot as plt

import numpy as np


class Quadratic:
    def __init__(
        self,
        n_points,
        a=2.0,
        b=-4.0,
        c=3.0,
        x_min=-10.0,
        x_max=10.0,
        noise_std=10.0
    ):
        self.X = np.linspace(
            x_min,
            x_max,
            n_points
        ).reshape(-1, 1)

        noise = (
            np.random.randn(
                n_points,
                1
            )
            * noise_std
        )

        self.Y = (
            a * self.X ** 2
            + b * self.X
            + c
            + noise
        )

        self.true_a = a
        self.true_b = b
        self.true_c = c


class DenseLayer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 1, keepdims = True)
        dinputs = np.dot(dvalues, self.weights.T)
        return dinputs
    
np.random.seed(0)
inputs = Quadratic(n_points=20)
plt.scatter(inputs.X, inputs.Y, s = 40)
plt.show()