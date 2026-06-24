import numpy as np
import matplotlib.pyplot as plt

class Spiral:
    def __init__(self, n_inputs, n_classes, n_dimensions):
        self.N = n_inputs
        self.K = n_classes
        self.D = n_dimensions
        self.P = np.zeros((self.N * self.K, self.D))
        self.L = np.zeros(self.N * self.K, dtype = 'uint8')

    def draw(self):
        for j in range(self.K):
            r_max = 2 * (j + 1)
            t = np.linspace(0, 1, self.N)
            r = r_max * t
            theta = 2 * 2 * np.pi * t + (j * 2 * np.pi) / self.K + np.random.randn(self.N) * 0.5
            
            ix = range(self.N * j, self.N * (j + 1))
            if self.D == 2:
                self.P[ix] = np.c_[r * np.cos(theta), r * np.sin(theta)]
            self.L[ix] = j

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons) 

    def forward(self, inputs):
        # Lưu lại inputs để tính đạo hàm
        self.inputs = inputsAAS
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Tính đạo hàm cho Weights và Biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Tính đạo hàm để truyền ngược lại cho lớp phía trước
        self.dinputs = np.dot(dvalues, self.weights.T)

class Optimizer_SGD:
    def __init__(self, eta):
        self.learning_rate = eta

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class Activation_ReLU:
    def forward(self, inputs):
        # Lưu lại inputs để dùng cho backward
        self.inputs = inputs
        self.output = np.maximum(inputs, 0)

    def backward(self, dvalues):
        # Sao chép dvalues từ lớp sau truyền tới
        self.dinputs = dvalues.copy()
        
        # Gradient bằng 0 tại những nơi input <= 0
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward gộp
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    # Backward Pass: (y_pred - y_true) / N
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        # Nếu nhãn y_true đang ở dạng số nguyên (0, 1, 2)
        if len(y_true.shape) == 1:
            y_true = np.eye(len(dvalues[0]))[y_true] # Chuyển thành dạng One-hot

        # Tính gradient: Dự đoán trừ đi nhãn thực tế
        self.dinputs = dvalues.copy()
        self.dinputs -= y_true
        
        # Chuẩn hóa gradient theo kích thước batch
        self.dinputs = self.dinputs / samples

class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        loss = np.mean(sample_losses)
        return loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_predictions, y_true):
        samples = len(y_predictions)

        y_predictions = np.clip(y_predictions, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_predictions[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_predictions * y_true, axis = 1)

        correct_confidence = -np.log(correct_confidence)
        return correct_confidence

class Loss_MeanSquareError(Loss):
    def forward(self, y_predictions, y_true):
        squared_error = (y_predictions - y_true) ** 2
        sample_losses = np.mean(squared_error, axis = 1)
        return sample_losses

N = 1000
K = 3
D = 2
epochs = 10000
eta = 0.2
inputs = Spiral(N, K, D)
inputs.draw()
x = inputs.P
y = inputs.L

plt.scatter(inputs.P[:, 0], inputs.P[:, 1], c=inputs.L, s = 40)
plt.show()

layer1 = Dense(n_inputs=2, n_neurons=64)
activation1 = Activation_ReLU()

layer2 = Dense(n_inputs=64, n_neurons=3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimize = Optimizer_SGD(eta)

for epoch in range(epochs):
    layer1.forward(inputs.P)
    activation1.forward(layer1.output)
    
    layer2.forward(activation1.output)
    loss = loss_activation.forward(layer2.output, y)

    # Tính Accuracy (Độ chính xác) để theo dõi
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # --- BACKWARD PASS ---
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # --- TỐI ƯU HÓA (UPDATE WEIGHTS) ---
    optimize.update_params(layer1)
    optimize.update_params(layer2)

    if epoch % 1000:
        plt.scatter(inputs.P[:, 0], inputs[:, 1], c = predictions, s = 40)
        print(f"Epoch: {epoch} | Loss: {loss:.3f} | Accuracy: {accuracy:.3f}")