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

class Optimizer_SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.iterations = 0

    def pre_update_params(self):
        self.learning_rate = self.learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1

class Optimizer_SGD_Decay:
    def __init__(self, learning_rate, decay_rate):
        self.intial_lr = learning_rate
        self.decay_rate = decay_rate
        self.current_lr = learning_rate
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = (self.intial_lr / (1 + self.decay_rate * self.iterations))

    def update_params(self, layer):
        self.weights -= self.current_lr * self.dweights
        self.biases -= self.current_lr * self.dbiases
    
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Momentum:
    def __init__(self, beta, learning_rate):
        self.learning_rate = learning_rate
        self.beta = beta
        self.weight_velocity = 0
        self.bias_velocity = 0
        self.iterations = 0

    def pre_update_params(self):
        self.learning_rate = self.learning_rate

    def update_params(self, layer):
        self.weight_velocity = self.beta * self.weight_velocity + self.learning_rate * layer.dweights
        self.bias_velocity = self.beta * self.bias_velocity + self.learning_rate * layer.dbiases

        layer.weights -= self.learning_rate * self.weight_velocity
        layer.biases -= self.learning_rate * self.bias_velocity

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Momentum_Decay:
    def __init__(self, beta, learning_rate, decay_rate):
        self.initial_lr = learning_rate
        self.decay_rate = decay_rate
        self.beta = beta
        self.weight_velocity = 0
        self.bias_velocity = 0
        self.iterations = 0

    def pre_update_params(self):
        self.learning_rate = self.initial_lr * 1 / (1 + self.decay_rate * self.iterations)

    def update_params(self, layer):
        self.weight_velocity = self.beta * self.weight_velocity + self.learning_rate * layer.dweights
        self.bias_velocity = self.beta * self.bias_velocity + self.learning_rate * layer.dbiases

        layer.weights -= self.learning_rate * self.weight_velocity
        layer.biases -= self.learning_rate * self.bias_velocity

    def post_update_params(self):
        self.iterations += 1

class Model:
    def __init__(self):
        self.layers = []

    # dùng để thêm layer và activation
    def add(self, layer):
        self.layers.append(layer)

    def set(self, loss, optimizer, accuracy = None):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    