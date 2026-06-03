import numpy as np

class DenseLayer:
    def __init__(self, n_neurons, n_inputs, activation_type = 'relu'):
        self.weights = 0.01 * np.random.randn((n_neurons, n_inputs))
        self.biases = np.zeros((1, n_neurons))
        self.activation_type = activation_type

    def _activation(self, z):
        if self.activation_type == 'relu':
            return np.maximum(z, 0) 
        elif self.activation_type == 'softmax':
            exp_values = np.exp(z - np.max(z, axis = 1, keepdims = True))
            return exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        
    def _activation_prime(self, z):
        if self.activation_type == 'relu':
            return np.where(z > 0, 1, 0)

    def forward(self, inputs):
        self.dinputs = inputs

        # sum
        self.output = np.dot(inputs, self.weights) + self.biases
        # activation function:
        self.A = self._activation(self.output)
        return self.A
    
    def backward(self, dA):
        if self.activation_type == 'softmax':
           sum_A_dA = np.sum(self.output * dA, axis = 1, keepdims = True)
           dZ = self.output * (dA - sum_A_dA)
        else:
            dZ = dA * self._activation_prime(self.output)
        
        self.dW = np.dot(self.inputs.T, dZ)
        self.db = np.sum(dZ, axis = 0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)
        return dX
    
    def update_weights(self, lr):
        self.weights -= lr * self.dW
        self.biases -= lr * self.db

class Loss_CategoricalCrossEntropy:
    def forward(self, inputs, y_true):
        samples = range(inputs)

        y_pred = np.clip(inputs, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            confidence_score = y_pred[range(samples), y_true]
        elif len(y_true.shape) == 2:
            confidence_score = np.sum(y_pred * y_true, axis = 1, keepdims = True)

        confidence_score = -np.log(confidence_score)
        return confidence_score

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

N = 1000
K = 3
D = 2
inputs = Spiral(n_points = N, n_classes = K, n_dimensions = D)

layer1 = DenseLayer(n_inputs = 2, n_neurons = 2)
layer2 = DenseLayer(n_inputs = 2, n_neurons = 3, activation_type = 'relu')
layer3 = DenseLayer(n_inputs = 3, n_neurons = 3, activation_type = 'softmax')

layer1.forward(inputs.P)
layer2.forward(layer1.A)
layer3.forward(layer2.A)

losses = Loss_CategoricalCrossEntropy()
loss = losses.forward(layer3.A, inputs.C)
print("Losses: %f", loss)