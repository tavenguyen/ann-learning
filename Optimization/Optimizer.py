import numpy as np

class Optimizer_SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dW
        layer.biases += -self.learning_rate * layer.dB

class Optimizer_SGD_Decay:
    def __init__(self, lr = 1.0, decay_rate = 0.0):
        self.initial_lr = lr
        self.current_lr = lr
        self.decay_rate = decay_rate
        self.iterations = 0
    
    def pre_update_params(self):
        self.current_lr = self.initial_lr * (1 / (1 + self.decay_rate * self.iterations))

    def update_params(self, layer):
        layer.weights += -self.current_lr * layer.dW
        layer.biases += -self.current_lr * layer.dB

    def post_update_params(self):
        self.iterations += 1

class Optimizer_SGD_Momentum:
    def __init__(self, lr = 1.0, beta = 0.0):
        self.initial_lr = lr
        self.current_lr = lr
        self.beta = beta
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = self.current_lr
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_velocity"):
            layer.weight_velocity = np.zeros_like(layer.weights)
            layer.bias_velocity = np.zeros_like(layer.biases)

        layer.weight_velocity = self.beta * layer.weight_velocity + self.current_lr * layer.dW
        layer.bias_velocity = self.beta * layer.bias_velocity + self.current_lr * layer.dB

        layer.weights -= layer.weight_velocity
        layer.biases -= layer.bias_velocity

    def post_update_params(self):
        self.iterations += 1

class Optimizer_SGD_Momentum_Decay:
    def __init__(self, lr = 1.0, decay_rate = 0.0, beta = 0.0):
        self.initial_lr = lr
        self.current_lr = lr
        self.decay_rate = decay_rate
        self.beta = beta
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = self.initial_lr * (1 / (1 + self.decay_rate * self.iterations))

    def post_update_params(self):
        self.iterations += 1
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_velocity"):
            layer.weight_velocity = np.zeros_like(layer.weights)
            layer.bias_velocity = np.zeros_like(layer.biases)

        # v_{i + 1} = \beta \cdot v_{i} + lr * dw
        layer.weight_velocity = self.beta * layer.weight_velocity + self.current_lr * layer.dW
        layer.bias_velocity = self.beta * layer.bias_velocity + self.current_lr * layer.dB

        # W_{i + 1} = W_{i} - V_{i + 1}
        layer.weights -= layer.weight_velocity
        layer.biases -= layer.bias_velocity

class Optimizer_AdaGrad:
    def __init__(self, lr = 1.0, epsilon = 1e-7):
        self.initial_lr = lr
        self.current_lr = lr
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = self.current_lr
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dW ** 2
        layer.bias_cache += layer.dB ** 2

        layer.weights -= (self.current_lr / (np.sqrt(layer.weight_cache) + self.epsilon)) * layer.dW
        layer.biases -= (self.current_lr / (np.sqrt(layer.bias_cache) + self.epsilon)) * layer.dB

    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSProp:
    def __init__(self, lr = 1.0, p = 0.9, epsilon = 1e-7):
        self.initial_lr = lr
        self.current_lr = lr
        self.p = p
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = self.current_lr
    
    def update_params(self, layer):
        if not hasattr(layer, "moving_average_weight"):
            layer.moving_average_weight = np.zeros_like(layer.weights)
            layer.moving_average_bias = np.zeros_like(layer.biases)
        
        layer.moving_average_weight = self.p * layer.moving_average_weight + (self.dW ** 2) * (1 - self.p)
        layer.moving_average_bias = self.p * layer.moving_average_bias + (self.dB ** 2) * (1 - self.p)

        layer.weights -= self.current_lr * self.dW / (np.sqrt(layer.moving_average_weight) + self.epsilon)
        layer.biases -= self.current_lr * self.dB / (np.sqrt(layer.moving_average_bias) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1