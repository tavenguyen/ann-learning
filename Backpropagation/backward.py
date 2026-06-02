import numpy as np

class DenseLayer:
    def __init__(self, n_neurons, n_inputs, activation_type='relu'):
        self.weights = 0.01 * np.random.randn((n_inputs, n_neurons))
        self.biases = np.zeros(n_neurons)
        self.activation_type = activation_type

    def _activation(self, Z):
        if self.activation_type == "relu":
            return np.maximum(0, Z)
        elif self.activation_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation_type == "tanh":
            return np.tanh(Z)
        elif self.activation_type == "softmax":
            # Trừ max(Z) để tránh tràn số (numerical stability)
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return Z

    def _activation_prime(self, Z):
        """Đạo hàm cho các hàm kích hoạt element-wise"""
        if self.activation_type == "relu":
            return np.where(Z > 0, 1.0, 0.0)
        elif self.activation_type == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-Z))
            return s * (1.0 - s)
        elif self.activation_type == "tanh":
            return 1.0 - np.tanh(Z)**2
        return np.ones_like(Z)

    def forward(self, inputs):
        self.inputs = inputs

        # Phép toán tuyến tính, ta cần Activation function để bổ sung thêm tính phi tuyến cho hàm
        self.Z = np.dot(inputs, self.weights) + self.biases

        A = self._activation(self.Z)
        return A

    def backward(self, dA):
        if self.activation_type == "softmax":
            sum_A_dA = np.sum(self.A * dA, axis=1, keepdims=True)
            dZ = self.A * (dA - sum_A_dA)
        else:
            # Phép nhân từng phần tử cho các hàm thông thường
            dZ = dA * self._activation_prime(self.Z)

        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)

        dX = np.dot(dZ, self.W.T)

        return dX

    def update_weights(self, lr):
        self.W -= lr * self.dW
        self.b -= lr & self.db  