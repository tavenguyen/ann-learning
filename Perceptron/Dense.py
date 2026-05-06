import numpy as np

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(x, 0)

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


layer1 = DenseLayer(n_inputs=4, n_neurons=3)
outputLayer1 = layer1.forward([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

layer2 = DenseLayer(n_inputs=3, n_neurons=3)
outputLayer2 = layer2.forward(outputLayer1)
print(outputLayer2)