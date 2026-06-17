import numpy as np
import matplotlib.pyplot as plt

class Quadratic:
    def __init__(self, n_points, n_classes, n_dimensions, noise_std = 2.0):
        N = n_points
        K = n_classes
        D = n_dimensions

        self.P = np.zeros((N * K, D))
        self.L = np.zeros(N * K, dtype = 'int8')

        for j in range(K):
            a = j + 1
            b = 2 * (j - 1)
            c = (j - 2) * 2
            noise = np.random.randn(N) * noise_std

            ix = range(N * j, N * (j + 1))
            t = np.linspace(-10, 10, N)
            if D == 2:
                self.P[ix] = np.c_[t, a * (t ** 2) + b * t + c + noise]
            self.L[ix] = j

class Optimizer_Adam:
    def __init__(
        self,
        learning_rate=0.001,
        decay_rate=0.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    ):
        self.initial_learning_rate = (
            learning_rate
        )

        self.current_learning_rate = (
            learning_rate
        )

        self.decay_rate = decay_rate

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.epsilon = epsilon

        self.iterations = 0

    def pre_update_params(self):
        self.current_learning_rate = (
            self.initial_learning_rate
            / (
                1
                + self.decay_rate
                * self.iterations
            )
        )

    def update_params(self, layer):
        if not hasattr(
            layer,
            "weight_momentums"
        ):
            layer.weight_momentums = (
                np.zeros_like(
                    layer.weights
                )
            )

            layer.bias_momentums = (
                np.zeros_like(
                    layer.biases
                )
            )

            layer.weight_cache = (
                np.zeros_like(
                    layer.weights
                )
            )

            layer.bias_cache = (
                np.zeros_like(
                    layer.biases
                )
            )

        layer.weight_momentums = (
            self.beta_1
            * layer.weight_momentums
            + (1 - self.beta_1)
            * layer.dweights
        )

        layer.bias_momentums = (
            self.beta_1
            * layer.bias_momentums
            + (1 - self.beta_1)
            * layer.dbiases
        )

        layer.weight_cache = (
            self.beta_2
            * layer.weight_cache
            + (1 - self.beta_2)
            * layer.dweights ** 2
        )

        layer.bias_cache = (
            self.beta_2
            * layer.bias_cache
            + (1 - self.beta_2)
            * layer.dbiases ** 2
        )

        weight_momentums_corrected = (
            layer.weight_momentums
            / (
                1
                - self.beta_1
                ** (self.iterations + 1)
            )
        )

        bias_momentums_corrected = (
            layer.bias_momentums
            / (
                1
                - self.beta_1
                ** (self.iterations + 1)
            )
        )

        weight_cache_corrected = (
            layer.weight_cache
            / (
                1
                - self.beta_2
                ** (self.iterations + 1)
            )
        )

        bias_cache_corrected = (
            layer.bias_cache
            / (
                1
                - self.beta_2
                ** (self.iterations + 1)
            )
        )

        layer.weights -= (
            self.current_learning_rate
            * weight_momentums_corrected
            / (
                np.sqrt(
                    weight_cache_corrected
                )
                + self.epsilon
            )
        )

        layer.biases -= (
            self.current_learning_rate
            * bias_momentums_corrected
            / (
                np.sqrt(
                    bias_cache_corrected
                )
                + self.epsilon
            )
        )

    def post_update_params(self):
        self.iterations += 1

class Optimizer_SGD:
    def __init__(self, lr = 0.001):
        self.learning_rate = lr
        self.iterations = 0

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    
    # Đạo hàm của activation linear = 1 nên lấy dvalues vào
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Loss:
    def calculate(self, y_pred, y_true):
        samples = self.forward(y_pred, y_true)
        self.output = np.mean(samples)
        return self.output
    
class Loss_MeanSquaredError:
    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        loss = np.mean(squared_error, axis = 1)
        return loss

    # error = 1/N \sum_M (y_pred - y_true)^2
    def backward(self, y_pred, y_true):
        N = y_pred.shape[0]
        M = y_pred.shape[1]
        
        self.dinputs = 2 * (y_pred - y_true) / (N * M)
        return self.dinputs

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

inputs = Quadratic(n_points=10, n_classes=1, n_dimensions=2)
layer1 = DenseLayer(n_inputs=1, n_neurons=8)
layer2 = DenseLayer(n_inputs=8, n_neurons=1)
loss_function = 
