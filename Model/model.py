import numpy as np

class DenseLayer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        return inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs
    
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, self.inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Model:
    def __init__(self):
        self.layers = []

    # dùng để thêm layer và activation
    def add(self, layer):
        self.layers.append(layer)
    