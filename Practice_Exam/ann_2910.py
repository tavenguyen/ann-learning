import numpy as np
import matplotlib.pyplot as plt

class SpiralData:
    def __init__(self, n_points, n_classes):
        dimensions = 2
        self.P = np.zeros((n_points * n_classes, dimensions))
        self.L = np.zeros(n_points * n_classes, dtype = 'uint8')

        for j in range(n_classes):
            ix = range(n_points * j , n_points * (j + 1))
            r = np.linspace(0, 1, n_points)
            theta = np.linspace(j * 4, (j + 1) * 4, n_points) + np.random.randn(n_points) * 0.6
        
            self.P[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
            self.L[ix] = j

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

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        loss = np.mean(sample_losses)
        return loss

class Activation_Softmax_Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        exp_values = np.exp(y_pred - np.max(y_pred, axis = 1, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        probabilities = self.output

        # loss
        samples = len(probabilities)
        prob_clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_conf = prob_clipped[range(samples), y_true]
        else:
            correct_conf = np.sum(prob_clipped * y_true)

        negative_log = -np.log(correct_conf)
        return negative_log

    def backward(self, y_pred, y_true):
        self.dinputs = y_pred.copy()
        samples = len(y_pred)

        # dZ = A - Y
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis = 1)

        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

class Accuracy:
    def calculate(self, y_pred, y_true):
        comparisions = self.compare(y_pred, y_true)
        accuracy = np.mean(comparisions)
        return accuracy
    
class Accuracy_CategoricalClassification(Accuracy):
    def compare(self, y_pred, y_true):
        if y_pred.ndim == 2:
            y_pred = np.argmax(y_pred, axis = 1)
        
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis = 1)

        return (y_pred == y_true)

class Layer_DropOut:
    def __init__(self, dropout_rate):
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("dropout phải thuộc khoảng [0, 1].")    
        
        self.dropout_rate = dropout_rate
        self.keep_probability = 1.0 - dropout_rate

    def forward(self, inputs, training = True):
        self.inputs = inputs
        if training is not True:
            self.output = inputs.copy()
            return
        
        self.binary_mask = np.random.binomial(1, self.keep_probability, size = inputs.shape) / self.keep_probability
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

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

n_classes = 3
n_points = 400
Training_Data = SpiralData(n_points = n_points, n_classes = n_classes)
X = Training_Data.P
Y = Training_Data.L

layer1 = Dense(n_inputs = 2, n_neurons = 32)
activation1 = Activation_ReLU()
dropout_layer1 = Layer_DropOut(dropout_rate=0.05)

layer2 = Dense(n_inputs= 32, n_neurons = 32)
activation2 = Activation_ReLU()
dropout_layer2 = Layer_DropOut(dropout_rate=0.05)

layerOutput = Dense(n_inputs = 32, n_neurons = 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
accuracy_function = Accuracy_CategoricalClassification()

momentum = 0.4
learning_rate = 0.001
decay = 1e-5
epsilon = 1e-7
beta1 = 0.9
beta2 = 0.99
optimizer = Optimizer_Adam(learning_rate=learning_rate, decay_rate=decay, epsilon=epsilon, beta1=beta1, beta2=beta2)

for epoch in range(10001):
    layer1.forward(X)
    activation1.forward(layer1.output)
    dropout_layer1.forward(activation1.output)

    layer2.forward(dropout_layer1.output)
    activation2.forward(layer2.output)
    dropout_layer2.forward(activation2.output)

    layerOutput.forward(dropout_layer2.output)

    loss = loss_activation.calculate(layerOutput.output, Y)
    predictions = loss_activation.output

    if len(Training_Data.L.shape) == 2:
        Training_Data.L = np.argmax(Training_Data.L, axis = 1)

    accuracy = accuracy_function.calculate(predictions, Y)

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, LR: {optimizer.current_lr}')

    loss_activation.backward(loss_activation.output, Y)
    layerOutput.backward(loss_activation.dinputs)
    dropout_layer2.backward(layerOutput.dinputs)
    activation2.backward(dropout_layer2.dinputs)
    layer2.backward(activation2.dinputs)
    dropout_layer1.backward(layer2.dinputs)
    activation1.backward(dropout_layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.update_params(layerOutput)
    optimizer.post_update_params()

TestZone = SpiralData(n_points=50, n_classes=3)
layer1.forward(TestZone.P)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layerOutput.forward(activation2.output)
predictions = np.argmax(layerOutput.output, axis=1)

plt.figure(figsize=(10, 4))

# Đồ thị 1: Dữ liệu Test thực tế
plt.subplot(1, 2, 1)
plt.title("Dữ liệu Test Thực tế")
plt.scatter(TestZone.P[:, 0], TestZone.P[:, 1], c=TestZone.L, cmap='brg', s=5)

# Đồ thị 2: Dữ liệu do Mô hình Dự đoán
plt.subplot(1, 2, 2)
plt.title("Mô hình Dự đoán")
plt.scatter(TestZone.P[:, 0], TestZone.P[:, 1], c=predictions, cmap='brg', s=5)

plt.show()