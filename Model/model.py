import numpy as np

class DenseLayer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

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
        return self.dinputs

class Optimizer_SGD:
    def __init__(self, learning_rate = 0.01):
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
    def __init__(self, learning_rate = 0.01, decay_rate = 1e-3):
        self.initial_lr = learning_rate
        self.decay_rate = decay_rate
        self.current_lr = learning_rate
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = (self.initial_lr / (1 + self.decay_rate * self.iterations))

    def update_params(self, layer):
        layer.weights -= self.current_lr * layer.dweights
        layer.biases -= self.current_lr * layer.dbiases
    
    def post_update_params(self):
        self.iterations += 1

class Optimizer_SGD_Momentum:
    def __init__(self, beta = 0.9, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.beta = beta
        self.iterations = 0

    def pre_update_params(self):
        self.learning_rate = self.learning_rate

    def update_params(self, layer):
        if not hasattr(layer, "weight_velocity"):
            layer.weight_velocity = np.zeros_like(layer.weights)
            layer.bias_velocity = np.zeros_like(layer.biases)

        layer.weight_velocity = self.beta * layer.weight_velocity + self.learning_rate * layer.dweights
        layer.bias_velocity = self.beta * layer.bias_velocity + self.learning_rate * layer.dbiases

        layer.weights -= layer.weight_velocity
        layer.biases -= layer.bias_velocity

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Momentum_Decay:
    def __init__(self, beta = 0.9, learning_rate = 0.01, decay_rate = 1e-3):
        self.initial_lr = learning_rate
        self.decay_rate = decay_rate
        self.beta = beta
        self.iterations = 0

    def pre_update_params(self):
        self.learning_rate = self.initial_lr * 1 / (1 + self.decay_rate * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, "weight_velocity"):
            layer.weight_velocity = np.zeros_like(layer.weights)
            layer.bias_velocity = np.zeros_like(layer.biases)

        layer.weight_velocity = self.beta * layer.weight_velocity + self.learning_rate * layer.dweights
        layer.bias_velocity = self.beta * layer.bias_velocity + self.learning_rate * layer.dbiases

        # velocity already includes learning rate, apply directly
        layer.weights -= layer.weight_velocity
        layer.biases -= layer.bias_velocity

    def post_update_params(self):
        self.iterations += 1

class Optimizer_AdaGrad:
    def __init__(self, learning_rate = 0.01, epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.iterations = 0
    
    def pre_update_params(self):
        self.learning_rate = self.learning_rate

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        layer.weights -= (self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon))
        layer.biases -= (self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon))

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSProp:
    def __init__(self, learning_rate = 0.001, p = 0.9, epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.iterations = 0
        # hệ số giữ lại thông tin
        self.p = p
        self.epsilon = epsilon

    def pre_update_params(self):
        self.learning_rate = self.learning_rate
    
    def update_params(self, layer):
        if not hasattr(layer, "moving_average_weight"):
            layer.moving_average_weight = np.zeros_like(layer.weights)
            layer.moving_average_bias = np.zeros_like(layer.biases)

        layer.moving_average_weight = self.p * layer.moving_average_weight + (1 - self.p) * layer.dweights ** 2
        layer.moving_average_bias = self.p * layer.moving_average_bias + (1 - self.p) * layer.dbiases ** 2

        layer.weights -= self.learning_rate * layer.dweights / (np.sqrt(layer.moving_average_weight) + self.epsilon)
        layer.biases -= self.learning_rate * layer.dbiases / (np.sqrt(layer.moving_average_bias) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7):
        self.current_lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = self.current_lr

    def update_params(self, layer):
        if not hasattr(layer, "cache_weight"):
            layer.cache_weight_g = np.zeros_like(layer.weights)
            layer.cache_weight_g2 = np.zeros_like(layer.weights)
            layer.cache_bias_g = np.zeros_like(layer.biases)
            layer.cache_bias_g2 = np.zeros_like(layer.biases)

        layer.cache_weight_g = self.beta1 * layer.cache_weight_g + (1 - self.beta1) * layer.dweights
        layer.cache_weight_g2 = self.beta2 * layer.cache_weight_g2 + (1 - self.beta2) * layer.dweights ** 2

        layer.cache_bias_g = self.beta1 * layer.cache_bias_g + (1 - self.beta1) * layer.dbiases
        layer.cache_bias_g2 = self.beta2 * layer.cache_bias_g2 + (1 - self.beta2) * layer.dbiases ** 2

        layer.cache_weight_g_correction = layer.cache_weight_g / (1 - self.beta1 ** (self.iterations + 1))
        layer.cache_weight_g2_correction = layer.cache_weight_g2 / (1 - self.beta2 ** (self.iterations + 1))

        layer.cache_bias_g_correction = layer.cache_bias_g / (1 - self.beta1 ** (self.iterations + 1))
        layer.cache_bias_g2_correction = layer.cache_bias_g2 / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights -= self.current_lr * layer.cache_weight_g_correction / (np.sqrt(layer.cache_weight_g2_correction) + self.epsilon)
        layer.biases -= self.current_lr * layer.cache_bias_g_correction / (np.sqrt(layer.cache_bias_g2_correction) + self.epsilon)

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

    