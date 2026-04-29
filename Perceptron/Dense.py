import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


layer1 = DenseLayer(n_inputs=4, n_neurons=3)
outputLayer1 = layer1.forward([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

layer2 = DenseLayer(n_inputs=3, n_neurons=3)
outputLayer2 = layer2.forward(outputLayer1)
print(outputLayer2)