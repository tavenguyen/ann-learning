import numpy as np
import matplotlib.pyplot as plt

class PointZone:
    def __init__(self, n_points, n_classes, cluster_std=1.0, radius=3.0, random_state=None):
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        self.P = np.zeros((n_points * n_classes, 2))
        self.L = np.zeros(n_points * n_classes, dtype='uint8')
        
        for j in range(n_classes):
            offset_x = np.cos(j * np.pi / 2) * radius
            offset_y = np.sin(j * np.pi / 2) * radius
            
            x = rng.normal(loc=offset_x, scale=cluster_std, size=n_points)
            y = rng.normal(loc=offset_y, scale=cluster_std, size=n_points)
            
            ix = range(n_points * j, n_points * (j + 1))
            self.P[ix] = np.c_[x, y]
            self.L[ix] = j

class Dense:
    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output,single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs

class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        self.output = np.mean(sample_losses)
        return self.output
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_conf = y_pred_clipped[range(sample), y_true]
        else:
            correct_conf = np.sum(y_pred_clipped * y_true, axis = 1)
        
        negative_log_likelihood = -np.log(correct_conf)
        return negative_log_likelihood
        
    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        labels = len(y_pred[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / y_pred
        self.dinputs /= samples
            
class Activation_Softmax_Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        exp_values = np.exp(y_pred - np.max(y_pred, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

        # loss calculate
        samples = len(probabilities)
        y_pred_clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_conf = y_pred_clipped[range(samples), y_true]
        else:
            correct_conf = np.sum(y_pred_clipped * y_true, axis = 1)
        
        negative_log_likelihood = -np.log(correct_conf)
        return negative_log_likelihood


    def backward(self, y_pred, y_true):
        samples = len(y_pred)

        # dZ = A - Y
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis = 1)

        y_pred[range(samples), y_true] -= 1
        self.dinputs = y_pred / samples
        return self.dinputs

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

Zone = PointZone(n_points=800, n_classes=4)

X_input = Zone.P      # Shape: (3200, 2) - Dữ liệu có 2 neuron ngõ vào
Y_true = Zone.L       # Shape: (3200,)   - Nhãn (Class từ 0 đến 3)
EPOCHS = 10001

plt.scatter(X_input[:, 0], X_input[:, 1], s=5, c=Y_true, cmap='brg')
plt.show()

dense_layer1 = Dense(n_inputs=2, n_neurons=32)
activation1 = Activation_ReLU()
dense_layer2 = Dense(n_inputs=32, n_neurons=32)
activation2 = Activation_ReLU()
layerOutput = Dense(n_inputs=32, n_neurons=4) # Output 4 class

loss_function = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.001)

for epoch in range(EPOCHS):
    dense_layer1.forward(X_input)
    activation1.forward(dense_layer1.output)
    dense_layer2.forward(activation1.output)
    activation2.forward(dense_layer2.output)
    layerOutput.forward(activation2.output)

    loss_function.calculate(layerOutput.output, Y_true)
    loss = loss_function.output

    loss_function.backward(layerOutput.output, Y_true)
    layerOutput.backward(loss_function.dinputs)
    activation2.backward(layerOutput.dinputs)
    dense_layer2.backward(activation2.dinputs)
    activation1.backward(dense_layer2.dinputs)
    dense_layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense_layer1)
    optimizer.update_params(dense_layer2)
    optimizer.update_params(layerOutput)
    optimizer.post_update_params()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

TestZone = PointZone(n_points=800, n_classes=4)
X_test = TestZone.P         # Tọa độ điểm Test (400, 2)
Y_test_true = TestZone.L    # Nhãn thực tế để đối chiếu (400,)

# 2. Quá trình DỰ ĐOÁN (Chỉ có Forward Pass, KHÔNG CÓ Backward/Optimizer)
dense_layer1.forward(X_test)
activation1.forward(dense_layer1.output)
dense_layer2.forward(activation1.output)
activation2.forward(dense_layer2.output)
layerOutput.forward(activation2.output)

# layerOutput.output lúc này có shape là (3200, 4)
# 3. Chuyển xác suất (3200, 4) thành Nhãn dự đoán (3200,) bằng argmax
predictions = np.argmax(layerOutput.output, axis=1)

plt.figure(figsize=(10, 4))

# Đồ thị 1: Dữ liệu Test thực tế
plt.subplot(1, 2, 1)
plt.title("Dữ liệu Test Thực tế")
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test_true, cmap='brg', s=5)

# Đồ thị 2: Dữ liệu do Mô hình Dự đoán
plt.subplot(1, 2, 2)
plt.title("Mô hình Dự đoán")
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='brg', s=5)

plt.show()