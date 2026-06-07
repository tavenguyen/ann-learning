import numpy as np
import matplotlib.pyplot as plt

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

class DenseLayer:
    def __init__(self, n_neurons, n_inputs, activation_type = 'linear'):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation_type = activation_type

    def _activation(self, Z):
        if self.activation_type == 'relu':
            return np.maximum(Z, 0)
        return Z
    
    def _activation_prime(self, Z):
        if self.activation_type == 'relu':
            return np.where(Z > 0, 1, 0)
        return np.ones_like(Z)

    def forward(self, inputs):
        self.inputs = inputs

        self.output = np.dot(inputs, self.weights) + self.biases
        self.A = self._activation(self.output)
        return self.A

    def backward(self, dA):
        dZ = dA * self._activation_prime(self.output)
        self.dW = np.dot(self.inputs.T, dZ)
        self.dB = np.sum(dZ, axis = 0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)
        return dX
    
    def update_weights(self, lr):
        self.weights = self.weights - lr * self.dW
        self.biases = self.biases - lr * self.dB

class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        loss = np.mean(sample_losses)
        return loss    

class Activation_Softmax_Loss_CrossCategoricalEntropy(Loss):
    def forward(self, inputs, y_true):
        # softmax activation
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        self.output = exp_values / (np.sum(exp_values, axis = 1, keepdims=True))

        # loss
        sample = len(inputs)
        y_pred = np.clip(self.output, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            confidence_score = y_pred[range(sample), y_true]
        elif len(y_true.shape) == 2:
            confidence_score = np.sum(y_pred * y_true, axis = 1)
        
        confidence_score = -np.log(confidence_score)
        return confidence_score
    
    def backward(self, dvalues, y_true):
        # dvalues là softmax activation
        samples = len(dvalues)

        self.dinputs = dvalues.copy()
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        # dZ = A - Y
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples
        return self.dinputs
    
def learning_rate_decay(initial_lr, decay_step, decay_rate):
    return initial_lr * (1 / (1 + decay_rate * decay_step))

N = 100
D = 2
K = 3
initial_lr = 1
decay_rate = 0.5
EPOCHS = 10000

inputs = Spiral(n_points=N, n_classes=K, n_dimensions=D)
layer1 = DenseLayer(n_inputs=2, n_neurons=64, activation_type='relu')
layer2 = DenseLayer(n_inputs=64, n_neurons=3)
loss_activation = Activation_Softmax_Loss_CrossCategoricalEntropy()

plt.scatter(inputs.P[:, 0], inputs.P[:, 1], c = inputs.L, s = 40)
plt.show()

lr = initial_lr
for epoch in range(EPOCHS):
    layer1.forward(inputs.P)
    layer2.forward(layer1.A)

    # loss calculate
    loss = loss_activation.calculate(layer2.A, inputs.L)

    # accuracy calculate
    predictions = np.argmax(loss_activation.output, axis = 1)
    accuracy = np.mean(predictions == inputs.L)

    if epoch % 1000 == 0:
        plt.scatter(inputs.P[:, 0], inputs.P[:, 1], c = predictions, s = 40)
        plt.show()
        print("Losses: ", loss, "- Accuracy: ", accuracy)

    # backward
    loss_activation.backward(loss_activation.output, inputs.L)
    dX_2 = layer2.backward(loss_activation.dinputs)
    dX_1 = layer1.backward(dX_2)

    # lr = learning_rate_decay(initial_lr=initial_lr, decay_step=epoch, decay_rate=decay_rate)

    layer1.update_weights(lr)
    layer2.update_weights(lr)
    