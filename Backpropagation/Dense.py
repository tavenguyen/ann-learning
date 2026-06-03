import numpy as np

class DenseLayer:
    def __init__(self, n_neurons, n_inputs, activation_type = 'linear'):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation_type = activation_type

    def _activation(self, z):
        if self.activation_type == 'relu':
            return np.maximum(z, 0) 
        return z
        
    def _activation_prime(self, z):
        if self.activation_type == 'relu':
            return np.where(z > 0, 1, 0)
        return np.ones_like(z) # Đạo hàm linear là 1

    def forward(self, inputs):
        self.inputs = inputs

        # sum
        self.output = np.dot(inputs, self.weights) + self.biases
        # activation function:
        self.A = self._activation(self.output)
        return self.A
    
    def backward(self, dA):
        dZ = dA * self._activation_prime(self.output)
        
        self.dW = np.dot(self.inputs.T, dZ)
        self.db = np.sum(dZ, axis = 0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)
        return dX
    
    def update_weights(self, lr):
        self.weights -= lr * self.dW
        self.biases -= lr * self.db

class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        loss = np.mean(sample_losses)
        return loss

class Activation_Softmax_LossCategoricalCrossEntropy(Loss):
    def forward(self, inputs, y_true):
        # Tính activation softmax
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output =  exp_values / (np.sum(exp_values, axis = 1, keepdims = True))

        # Tính loss
        samples = len(inputs)
        y_pred = np.clip(self.output, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            confidence_score = y_pred[range(samples), y_true]
        elif len(y_true.shape) == 2:
            confidence_score = np.sum(y_pred * y_true, axis = 1)

        confidence_score = -np.log(confidence_score)
        return confidence_score
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        self.dinputs = dvalues.copy()
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        # dZ = A - Y
        self.dinputs[range(samples), y_true] -= 1 
        self.dinputs /= samples
        return self.dinputs
    

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

N = 1000
K = 3
D = 2
inputs = Spiral(n_points = N, n_classes = K, n_dimensions = D)

layer1 = DenseLayer(n_inputs = 2, n_neurons = 64, activation_type = 'relu')
layer2 = DenseLayer(n_inputs = 64, n_neurons = 3)
loss_activation = Activation_Softmax_LossCategoricalCrossEntropy()

EPOCHS = 10000
learning_rate = 1

for epoch in range(EPOCHS):
    # Forward pass
    layer1.forward(inputs.P)
    layer2.forward(layer1.A)

    # Loss calculate
    loss = loss_activation.calculate(layer2.A, inputs.L)

    # Accuracy calculate
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == inputs.L)

    if epoch % 1000 == 0:
        print("Losses: ", loss, "- Accuracy: ", accuracy)

    # --- BACKWARD PASS ---
    # Truyền Softmax Output và Nhãn vào để lấy dZ đầu tiên
    loss_activation.backward(loss_activation.output, inputs.L)

    # Lớp 2 nhận dZ từ lớp Fusion (loss_activation.dinputs), trả ra dX_2
    dX_2 = layer2.backward(loss_activation.dinputs)

    # Lớp 1 nhận dX_2 (coi như dA của nó), trả ra dX_1
    dX_1 = layer1.backward(dX_2)

    layer1.update_weights(learning_rate)
    layer2.update_weights(learning_rate)