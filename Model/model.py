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

#----------------------------- Activation -------------------------------#
class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.inputs
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs
    
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

#----------------------------- Loss -------------------------------#
class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        self.output = np.mean(sample_losses)
        return self.output
    
class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        # Softmax activation
        exp_values = np.exp(y_pred - np.max(y_pred, axis = 1, keepdims = True))
        probs = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probs
        
        samples = len(probs)
        # Clip probabilities to prevent log(0)
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)

        # y_true =[0, 1, 1]
        if len(y_true.shape) == 1:
            correct_conf = probs_clipped[range(samples), y_true]
        else:
            # y_true is one-hot
            correct_conf = np.sum(probs_clipped * y_true, axis = 1)

        negative_log_likelihood = -np.log(correct_conf)
        return negative_log_likelihood

    def backward(self, y_pred, y_true):
        # Backward API expects (y_pred, y_true) to match Model.backward
        # Use stored softmax probabilities computed in forward (self.output)
        probs = self.output.copy()
        samples = len(probs)

        # If y_true is one-hot encoded, convert to class indices
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        # Gradient for softmax combined with categorical cross-entropy
        probs[range(samples), y_true] -= 1
        self.dinputs = probs / samples
        return self.dinputs

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        error = np.sum(squared_error, axis = 1)
        return error

    def backward(self, y_pred, y_true):
        samples = y_pred.shape[0] 
        outputs = y_pred.shape[1]
        # y_pred, y_true co shape (N, M) voi N la so luong mau, M la so output trong moi mau

        self.dinputs = (2 * (y_pred - y_true)) / (outputs * samples)
        return self.dinputs

#----------------------------- Accuracy -------------------------------#
class Accuracy:
    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, y_pred, y_true):
        comparisions = self.compare(y_pred, y_true)
        accuracy = np.mean(comparisions)
        
        self.accumulated_sum += np.sum(comparisions)
        self.accumulated_count += comparisions.size

        return accuracy
    
    def calculate_accumulated(self):
        if self.accumulated_count == 0:
            return 0.0
        
        return (self.accumulated_sum / self.accumulated_count)
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Trước mỗi epoch: accuracy.new_pass()
# Sau mỗi epoch: batch_accuracy = accuracy.calculate(predictions, y_batch)
# Cuối epoch: epoch_accuracy = accuracy.calculate_accumulated()
class Accuracy_RegressionTolerance(Accuracy):
    def __init__(self, tolerance=None):
        self.tolerance = tolerance
        self.new_pass()

    def init(self, y):
        if self.tolerance is None:
            self.tolerance = (np.std(y)/ 250)

    def compare(self, predictions, y):
        if self.tolerance is None:
            raise RuntimeError(
                "Cần gọi init(y) trước khi tính Accuracy."
            )

        return (np.abs(predictions - y) <= self.tolerance)

class Accuracy_CategoricalClassification(Accuracy):
    # cột: class, hàng: số mẫu
    # predictions = [
    # [0.7, 0.2, 0.1], 
    # [0.5, 0.2, 0.3],
    #]
    def compare(self, predictions, y_true):
        if predictions.ndim == 2:
            predicted_classes = np.argmax(
                predictions,
                axis=1
            )
        else:
            predicted_classes = predictions

        # y_true = [
        # [1, 0, 0]
        # [1, 0, 0]
        # [0, 0, 1]
        # ]
        if y_true.ndim == 2:
            true_classes = np.argmax(
                y_true,
                axis=1
            )
        else:
            true_classes = y_true

        return (
            predicted_classes
            == true_classes
        )

#----------------------------- Optimizer-------------------------------#
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

    # dùng để gắn component cho model
    def set(self, loss, optimizer, accuracy = None):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # tìm các layer có thể train
    def finalize(self):
        self.trainable_layers = []
        for layer in self.layers:
            if hasattr(layer, "weights"):
                self.trainable_layers.append(layer)

    # Dùng training=True để sau này hỗ trợ Dropout hoặc BatchNorm.
    # Với layer không cần training, ta gọi bình thường.
    def forward(self, X, training = True):
        output = X
        for layer in self.layers:
            try:
                layer.forward(output, training = training)
            except(TypeError):
                layer.forward(output)
            
            output = layer.output
        
        return output
    
    def backward(self, y_pred, y_true):
        self.loss.backward(y_pred, y_true)
        dvalues =  self.loss.dinputs

        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs

    def update_params(self):
        self.optimizer.pre_update_params()

        # Optimizer chỉ update layer có weights.
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)

        self.optimizer.post_update_params()

    # forward -> loss -> accuracy -> backward -> update params -> print
    def train(self, x, y, epochs = 10000, print_every = 100):
        for epoch in range(epochs):
            y_pred = self.forward(x)

            loss = self.loss.calculate(y_pred, y)
            if self.accuracy is not None:
                accuracy = self.accuracy.calculate(y_pred, y)
            else:
                accuracy = None
            
            self.backward(y_pred, y)

            self.update_params()

            if epoch % print_every == 0:
                if accuracy is not None:
                    print(
                        f"epoch={epoch}, "
                        f"loss={loss:.6f}, "
                        f"accuracy={accuracy:.6f}"
                    )
                else:
                    print(
                        f"epoch={epoch}, "
                        f"loss={loss:.6f}"
                    )

    # dự đoán sau khi train, chỉ gồm forward. Khác với training gồm forward, backward, update
    def predict(self, inputs):
        return self.forward(inputs)
    
    # đo loss và accuracy trên validation test 
    def evaluate(self, X_validation, y_validation):
        y_pred = self.forward(X_validation)

        loss = self.loss.calculate(y_pred, y_validation)
        if self.accuracy is not None:
            accuracy = self.accuracy.calculate(y_pred, y_validation)
        else:
            accuracy = None
        
        return loss, accuracy
