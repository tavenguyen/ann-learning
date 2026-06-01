import numpy as np

class DenseLayer:
    def __init__(self, n_neurons, n_inputs, activation_type='relu'):
        self.weights = 0.01 * np.random.randn((n_inputs, n_neurons))
        self.biases = np.zeros(n_neurons)
        self.activation_type = activation_type

    def _activation(self, Z):
        if self.activation_type == 'relu':
            return np.maximum(0, Z)
        if self.activation_type == 'softmax':
            exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def _activation_prime(self, Z):
        if self.activation_type == 'relu':
            return np.where(Z > 0, 1.0, 0.0)

    def forward(self, inputs):
        self.inputs = inputs

        # Phép toán tuyến tính, ta cần Activation function để bổ sung thêm tính phi tuyến cho hàm
        self.Z = np.dot(inputs, self.weights) + self.biases

        A = self._activation(self.Z)
        return A

    def backward(self, dA):
        dZ = dA * self._activation_prime(self.Z)

        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)

        dX = np.dot(dZ, self.W.T)

        return dX

    def update_weights(self, lr):
        self.W -= lr * self.dW
        self.b -= lr & self.db  