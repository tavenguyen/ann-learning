import numpy as np

class Activation_Linear:
    def forward(self, Z):
        return Z
    
    def backward(self, Z):
        return np.ones_like(Z)
    
class Activation_Relu:
    def forward(self, Z):
        return np.maximum(0, Z)
    
    def backward(self, Z):
        return np.where(Z > 0, 1, 0)

class Regularization_L1:
    def __init__(self, strength = 0.0):
        self.strength = strength

    def forward(self, parameters):
        return self.strength * np.sum(np.abs(parameters))
    
    def backward(self, parameters):
        gradient = np.ones_like(parameters)

        gradient[parameters < 0] = -1
        return self.strength * gradient
    
class Regularization_L2:
    def __init__(self, strength):
        self.strength = strength

    def forward(self, parameters):
        return self.strength * np.sum(parameters ** 2)
    
    def backward(self, parameters):
        return 2 * self.strength * parameters

class Regularization_L1L2:
    def __init__(self, l1_strength = 0.0, l2_strength = 0.0):
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

    def forward(self, weights):
        l1_loss = self.l1_strength * np.sum(np.abs(weights))
        l2_loss = self.l2_strength * np.sum(weights ** 2)
        return l1_loss + l2_loss
    
    def backward(self, weights):
        l1_gradient = np.ones_like(weights)
        l1_gradient[weights < 0] = -1

        l1_gradient *= self.l1_strength

        l2_gradient = 2 * self.l2_strength * weights
        return l1_gradient + l2_gradient

class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def forward(self, inputs, y_true):
        # activation softmax
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output = exp_values / (np.sum(exp_values, axis = 1, keepdims = True))
    
        # loss
        samples = len(inputs)
        y_pred = np.clip(self.output, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            confidence_scores = y_pred[range(samples), y_true] 
        elif len(y_true.shape) == 2:
            confidence_scores = np.sum(y_pred * y_true, axis = 1)

        confidence_scores = -np.log(confidence_scores)
        return confidence_scores

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        self.dinputs = dvalues.copy()
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        # dZ = A - Y
        self.dinputs[range(samples), y_true] -= 1 
        self.dinputs /= samples
        return self.dinputs

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

class DenseLayer:   
    def __init__(self, n_neurons, n_inputs, weight_regularizer = None, bias_regularizer = None, activation = Activation_Linear(), Optimizer = Optimizer_SGD(), DropOut = None):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_regularizer = weight_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation = activation
        self.drop_out = DropOut
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        self.A = self.output
        if self.activation is not None:
            self.A = self.activation.forward(self.output)
        return self.A
    
    def backward(self, dA):
        dZ = dA * self.activation.backward(self.output)
        self.dweights = np.dot(self.inputs.T, dZ)
        self.dbiases = np.sum(dZ, axis=0,keepdims=True)
        dX = np.dot(dZ, self.weights.T)

        if self.weight_regularizer is not None:
            self.dweights += self.weight_regularizer.backward(self.weights)

        if self.bias_regularizer is not None:
            self.dbiases += self.bias_regularizer.backward(self.biases)

        return dX

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

# parameters
N = 1000
D = 2
K = 3    
EPOCHS = 10000
learning_rate = 1.0


# inputs -> layer1 (ReLU) -> layer2 (Softmax) -> Loss (Categorical Cross Entropy)
inputs = Spiral(n_points = N, n_classes = K, n_dimensions = D)
layer1 = DenseLayer(n_inputs=2, n_neurons=64, weight_regularizer=Regularization_L1(0.05), bias_regularizer=None, activation=Activation_Relu())
layer2 = DenseLayer(n_inputs=64,n_neurons=3, weight_regularizer=Regularization_L2(0.05))
Loss = Activation_Softmax_Loss_CategoricalCrossEntropy()

for epoch in range(EPOCHS):
    layer1.forward(inputs.P)
    layer2.forward(layer1.A)
    
    loss = Loss.forward(layer2.A, inputs.L)

    predictions = np.argmax(loss.output, axis = 1)
    accuracy = np.mean(predictions == inputs.L)

    # backward
    loss.backward(loss.output, inputs.L)
    dX2 = layer2.backward(loss.dinputs)
    dX1 = layer1.backward(dX1)
    
    # update_weights
    layer1.Optimizer.pre_update_params()
    layer1.Optimizer.update_params()
    layer1.Optimizer.post_update_params()

    layer2.Optimizer.pre_update_params()
    layer2.Optimizer.update_params()
    layer2.Optimizer.post_update_params()