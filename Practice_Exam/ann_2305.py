import numpy as np
import matplotlib.pyplot as plt

class Circle:
    def __init__(self, n_points, n_classes, n_dimensions, noise_std=2.0, radius_step=2.0, random_state=None):
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        N = n_points
        K = n_classes
        D = n_dimensions

        self.P = np.zeros((N * K, D))
        self.L = np.zeros(N * K, dtype='uint8')

        for j in range(K):
            base_r = radius_step * (j + 1)
            r_noise = base_r + rng.normal(scale=noise_std, size=N)

            theta = np.linspace(0, 2 * np.pi, N)
            idx = range(N * j, N * (j + 1))
            self.P[idx] = np.c_[r_noise * np.cos(theta), r_noise * np.sin(theta)]
            self.L[idx] = j

class Optimizer_SGD_Decay_Momentum:
    def __init__(self, beta=0.9, learning_rate=0.01, decay=1e-3):
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
        self.beta = beta
        self.decay_rate = decay
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = self.initial_lr * (1 / (1 + self.decay_rate * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_velocity"):
            layer.weight_velocity = np.zeros_like(layer.weights)
            layer.bias_velocity = np.zeros_like(layer.biases)
        
        layer.weight_velocity = self.beta * layer.weight_velocity + self.current_lr * layer.dweights
        layer.bias_velocity = self.beta * layer.bias_velocity + self.current_lr * layer.dbiases

        layer.weights -= layer.weight_velocity
        layer.biases -= layer.bias_velocity

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate = 0.001, beta1=0.9, beta2=0.999, epsilon = 1e-7, decay_rate = 5e-3):
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.iterations = 0

    def pre_update_params(self):
        if self.decay_rate:
            self.current_lr = self.initial_lr * (1 / (1 + self.decay_rate * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_momentum"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentum = self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.dweights
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights ** 2

        layer.bias_momentum = self.beta1 * layer.bias_momentum + (1 - self.beta1) * layer.dbiases
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases ** 2

        # correction
        layer.weight_momentum_cr = layer.weight_momentum / (1 - self.beta1 ** (self.iterations + 1))
        layer.weight_cache_cr = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))

        layer.bias_momentum_cr = layer.bias_momentum / (1 - self.beta1 ** (self.iterations + 1))
        layer.bias_cache_cr = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))

        # update weights
        layer.weights -= self.current_lr * layer.weight_momentum_cr / (np.sqrt(layer.weight_cache_cr) + self.epsilon)
        layer.biases -= self.current_lr * layer.bias_momentum_cr / (np.sqrt(layer.bias_cache_cr) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_DropOut:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.keep_probability = 1.0 - dropout_rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.keep_probability, size = inputs.shape) / self.keep_probability
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Loss:
    def calculate(self, y_pred, y_true):
        samples = self.forward(y_pred, y_true)
        loss = np.mean(samples)
        return loss

class Activation_Softmax_Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        exp_values = np.exp(y_pred  - np.max(y_pred, axis = 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities

        # loss
        samples = len(y_pred)
        prob_clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_conf = prob_clipped[range(samples), y_true]
        else:
            correct_conf = np.sum(prob_clipped * y_true, axis = 1)
        negative_log_likelihood = -np.log(correct_conf)
        return negative_log_likelihood
    
    def backward(self, y_pred, y_true):
        self.dinputs = y_pred.copy()
        samples = len(y_pred)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples
            
n_points = 1000
n_classes = 3
n_dimensions = 2

TrainingData = Circle(n_points, n_classes, n_dimensions, noise_std=2.0, radius_step=6.0)
X = TrainingData.P[:, 0]
Y = TrainingData.P[:, 1]

plt.figure(1)
plt.scatter(X, Y, s = 5, c = TrainingData.L, cmap = plt.cm.Spectral)
plt.xlabel("Feature X0")
plt.ylabel("Feature X1")
plt.title("Training data")
plt.show()

Dense1 = Dense(2, 64)
activation1 = Activation_ReLU()
Dense_dropout1 = Layer_DropOut(dropout_rate=0.1)
Dense2 = Dense(64, 64)
activation2 = Activation_ReLU()
Dense_dropout2 = Layer_DropOut(dropout_rate=0.1)
Dense3 = Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay_rate=0.001)

for epoch in range(10001):
    Dense1.forward(TrainingData.P)
    activation1.forward(Dense1.output)
    Dense_dropout1.forward(activation1.output)

    Dense2.forward(Dense_dropout1.output)
    activation2.forward(Dense2.output)
    Dense_dropout2.forward(activation2.output)

    Dense3.forward(Dense_dropout2.output)

    loss = loss_activation.calculate(Dense3.output, TrainingData.L)
    predictions = np.argmax(loss_activation.output, axis = 1)
    accuracy = np.mean(predictions == TrainingData.L)

    loss_activation.backward(loss_activation.output, TrainingData.L)
    Dense3.backward(loss_activation.dinputs)
    Dense_dropout2.backward(Dense3.dinputs)
    activation2.backward(Dense_dropout2.dinputs)
    Dense2.backward(activation2.dinputs)
    Dense_dropout1.backward(Dense2.dinputs)
    activation1.backward(Dense_dropout1.dinputs)
    Dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(Dense1)
    optimizer.update_params(Dense2)
    optimizer.update_params(Dense3)
    optimizer.post_update_params()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, learning rate: {optimizer.current_lr}')

plt.scatter(X, Y, c=predictions, s=10, cmap=plt.cm.Spectral)
plt.title("Data Trained by the Neuron Network")
plt.xlabel("Feature X0")
plt.ylabel("Feature X1")
plt.show()