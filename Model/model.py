import numpy as np
import pickle

class DenseLayer:
    """Fully-connected (dense) layer.

    Parameters
    ----------
    n_neurons : int
        Number of output neurons.
    n_inputs : int
        Number of input features.
    weight_regularizer : optional
        Instance of Regularization_L1, Regularization_L2, or Regularization_L1L2.
        If provided, DenseLayer.backward will add regularization gradients
        automatically and you can obtain the regularization loss via
        weight_regularizer.forward(layer.weights).
    bias_regularizer : optional
        Same as weight_regularizer but applied to biases.

    Usage notes
    -----------
    - To include regularization in the reported loss, add the return value of
      layer.weight_regularizer.forward(layer.weights) (and bias) to the
      supervised loss computed by the model.
    - Regularizer.backward(parameters) should return the gradient contribution
      for the given parameters; DenseLayer will add it to dweights/dbiases.
    """
    def __init__(self, n_neurons: int, n_inputs: int, weight_regularizer=None, bias_regularizer=None):
        self.nInputs = n_inputs
        self.nNeurons = n_neurons
        
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Regularizer objects (can be None)
        self.weight_regularizer = weight_regularizer
        self.bias_regularizer = bias_regularizer
    
    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        # Gradients from data loss
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        # Add regularization gradients if regularizers provided
        if self.weight_regularizer is not None:
            self.dweights += self.weight_regularizer.backward(self.weights)

        if self.bias_regularizer is not None:
            self.dbiases += self.bias_regularizer.backward(self.biases)

        return self.dinputs
    
    def getConfig(self):
        return {
            "className": "DenseLayer",
            "nInputs": self.nInputs,
            "nNeurons": self.nNeurons
        }
    
    def getParameters(self):
        return {
            "weights": self.weights,
            "biases": self.biases
        }
    
    def setParameters(self, parameters):
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]

class Regularization_L1:
    """L1 regularization helper.

    Parameters
    ----------
    strength : float
        Regularization factor (lambda) for L1. Typical small values like 1e-4.

    Methods
    -------
    forward(parameters) -> float
        Returns L1 loss scalar for the given parameters: strength * sum(|parameters|).
    backward(parameters) -> np.ndarray
        Returns the gradient contribution to be added to parameter gradients:
        strength * sign(parameters).
    """
    def __init__(self, strength: float = 0.0):
        self.strength = strength

    # calculate loss
    def forward(self, parameters: np.ndarray) -> float:
        return self.strength * np.sum(np.abs(parameters))
    
    # calculate gradient
    def backward(self, parameters: np.ndarray) -> np.ndarray:
        gradient = np.ones_like(parameters)
        gradient[parameters < 0] = -1
        return self.strength * gradient
    
class Regularization_L2:
    """L2 regularization helper.

    Parameters
    ----------
    strength : float
        Regularization factor for L2. Typical small values like 1e-4.

    Methods
    -------
    forward(parameters) -> float
        Returns L2 loss scalar: strength * sum(parameters ** 2).
    backward(parameters) -> np.ndarray
        Returns gradient contribution: 2 * strength * parameters.
    """
    def __init__(self, strength: float = 0.0):
        self.strength = strength

    # calculate loss
    def forward(self, parameters: np.ndarray) -> float:
        return self.strength * np.sum(parameters ** 2)
    
    # calculate gradient
    def backward(self, parameters: np.ndarray) -> np.ndarray:
        return 2 * self.strength * parameters

class Regularization_L1L2:
    """Combined L1+L2 regularization.

    Parameters
    ----------
    l1_strength : float
    l2_strength : float

    Methods
    -------
    forward(parameters) -> float
        Sum of L1 and L2 losses.
    backward(parameters) -> np.ndarray
        Sum of L1 and L2 gradient contributions.
    """
    def __init__(self, l1_strength: float = 0.0, l2_strength: float = 0.0):
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

    # calculate loss
    def forward(self, weights: np.ndarray) -> float:
        l1_loss = self.l1_strength * np.sum(np.abs(weights))
        l2_loss = self.l2_strength * np.sum(weights ** 2)
        return l1_loss + l2_loss
    
    # calculate gradient
    def backward(self, weights: np.ndarray) -> np.ndarray:
        l1_gradient = np.ones_like(weights)
        l1_gradient[weights < 0] = -1

        l1_gradient *= self.l1_strength

        l2_gradient = 2 * self.l2_strength * weights
        return l1_gradient + l2_gradient

# chi su dung trong training
class Layer_Dropout:
    def __init__(self, dropout_rate):
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("dropout_rate phải thuộc khoảng [0, 1).")

        self.dropout_rate = dropout_rate
        self.keep_probability = 1.0 - dropout_rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = (
            np.random.binomial(
                1,
                self.keep_probability,
                size=inputs.shape
            )
            / self.keep_probability
        )

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
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
    
    def getConfig(self):
        return {
            "className": "ActivationLinear"
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass
    
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
    
    def getConfig(self):
        return {
            "className": "ActivationReLU"
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

#----------------------------- Loss -------------------------------#
class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        self.output = np.mean(sample_losses)
        return self.output
    
class Activation_Softmax_Loss_CategoricalCrossEntropy(Loss):
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
    
    def getConfig(self):
        return {
            "className": "ActivationSoftmaxLossCategoricalCrossEntropy"
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

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
    
    def getConfig(self):
        return {
            "className": "LossMeanSquaredError"
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

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
    
    def getConfig(self):
        return {
            "className": "AccuracyRegressionTolerance",
            "tolerance": self.tolerance
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

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
    
    def getConfig(self):
        return {
            "className": "AccuracyCategoricalClassification",
            "tolerance": self.tolerance
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

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

    def getConfig(self):
        return {
            "className": "OptimizerSGD",
            "learningRate": self.learning_rate
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

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

    def getConfig(self):
        return {
            "className": "OptimizerDecay",
            "learningRate": self.initial_lr,
            "decayRate": self.decay_rate
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

class Optimizer_SGD_Momentum:
    def __init__(self, beta = 0.9, learning_rate = 0.01):
        self.learning_rate = learning_rate
        self.beta = beta
        self.iterations = 0

    def pre_update_params(self):
        pass

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

    def getConfig(self):
        return {
            "className": "OptimizerMomentum",
            "learningRate": self.initial_lr,
            "beta": self.beta
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

class Optimizer_Momentum_Decay:
    def __init__(self, beta = 0.9, learning_rate = 0.01, decay_rate = 1e-3):
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
        self.decay_rate = decay_rate
        self.beta = beta
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = self.initial_lr * 1 / (1 + self.decay_rate * self.iterations)

    def update_params(self, layer):
        if not hasattr(layer, "weight_velocity"):
            layer.weight_velocity = np.zeros_like(layer.weights)
            layer.bias_velocity = np.zeros_like(layer.biases)

        layer.weight_velocity = self.beta * layer.weight_velocity + self.current_lr * layer.dweights
        layer.bias_velocity = self.beta * layer.bias_velocity + self.current_lr * layer.dbiases

        # velocity already includes learning rate, apply directly
        layer.weights -= layer.weight_velocity
        layer.biases -= layer.bias_velocity

    def post_update_params(self):
        self.iterations += 1

    def getConfig(self):
        return {
            "className": "OptimizerMomentumDecay",
            "learningRate": self.initial_lr,
            "decayRate": self.decay_rate,
            "beta": self.beta
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

class Optimizer_AdaGrad:
    def __init__(self, learning_rate = 0.01, epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.iterations = 0
    
    def pre_update_params(self):
        pass

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

    def getConfig(self):
        return {
            "className": "OptimizerAdaGrad",
            "learningRate": self.learning_rate,
            "epsilon": self.epsilon
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

class Optimizer_RMSProp:
    def __init__(self, learning_rate = 0.001, p = 0.9, epsilon = 1e-7):
        self.initial_lr = learning_rate
        self.current_lr = self.current_lr
        self.iterations = 0
        # hệ số giữ lại thông tin
        self.p = p
        self.epsilon = epsilon

    def pre_update_params(self):
        pass

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
        
    def getConfig(self):
        return {
            "className": "OptimizerRMSProp",
            "learningRate": self.initial_lr,
            "p": self.p,
            "epsilon": self.epsilon
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

class Optimizer_Adam:
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7, decay_rate = 3e-5):
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.iterations = 0

    def pre_update_params(self):
        self.current_lr = self.initial_lr * (1 / (1 + self.decay_rate * self.iterations))

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

    def getConfig(self):
        return {
            "className": "OptimizerAdam",
            "learningRate": self.initial_lr,
            "decayRate": self.decay_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

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

    def get_regularization_loss(self) -> float:
        reg_loss = 0.0
        for layer in getattr(self, "trainable_layers", []):
            if getattr(layer, "weight_regularizer", None) is not None:
                reg_loss += layer.weight_regularizer.forward(layer.weights)
            if getattr(layer, "bias_regularizer", None) is not None:
                reg_loss += layer.bias_regularizer.forward(layer.biases)
            return reg_loss

    # forward -> loss -> accuracy -> backward -> update params -> print
    def train(self, x, y, epochs = 10000, print_every = 100):
        for epoch in range(epochs):
            y_pred = self.forward(x)

            data_loss = self.loss.calculate(y_pred, y)
            reg_loss = self.get_regularization_loss()
            total_loss = data_loss + reg_loss
            if self.accuracy is not None:
                accuracy = self.accuracy.calculate(y_pred, y)
            else:
                accuracy = None
            
            self.backward(y_pred, y)

            self.update_params()

            if epoch % print_every == 0:
                if accuracy is not None:
                    print(f"epoch={epoch}, loss={total_loss:.6f}, data_loss={data_loss:.6f}, reg_loss={reg_loss:.6f}, accuracy={accuracy:.6f}")
                else:
                    print(f"epoch={epoch}, loss={total_loss:.6f}, data_loss={data_loss:.6f}, reg_loss={reg_loss:.6f}")

    # dự đoán sau khi train, chỉ gồm forward. Khác với training gồm forward, backward, update
    def predict(self, inputs):
        return self.forward(inputs)
    
    # đo loss và accuracy trên validation test 
    def evaluate(self, X_validation, y_validation):
        y_pred = self.forward(X_validation)

        data_loss = self.loss.calculate(y_pred, y_validation)
        reg_loss = self.get_regularization_loss()
        if self.accuracy is not None:
            accuracy = self.accuracy.calculate(y_pred, y_validation)
        else:
            accuracy = None
        
        return (data_loss + reg_loss), accuracy

    def getConfig(self):
        config = {
            "layers": [],
            "loss": None,
            "optimizer": None,
            "accuracy": None
        }   

        for layer in self.layers:
            config["layers"].append(layer.getConfig())

        if self.loss is not None:
            config["loss"] = self.loss.getConfig()

        if self.optimizer is not None:
            config["optimizer"] = self.optimizer.getConfig()

        if self.accuracy is not None:
            config["accuracy"] = self.accuracy.getConfig()

        return config

class DataLoader:
    def __init__(self, X, y, batch_size=32, suffle=True, drop_last=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.suffle = suffle
        self.drop_last = drop_last

        self.n_samples = len(X)
    
    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.n_samples)
        else:
            indices = np.arange(self.n_samples)
            
        X_shuffled = self.X[indices]
        y_shuffled = self.y[indices]

        for start in range(0, self.n_samples, self.batch_size):
            end = start + self.batch_size
            if end > self.n_samples and self.drop_last:
                return
            
            # cú pháp slicing sẽ tự dùng ở phần tử cuối cùng nếu end lớn hơn n_samples
            X_batches = self.X_shuffled[start:end]
            Y_batches = self.y_shuffled[start:end]
            yield X_batches, Y_batches

    def __len__(self):
        if self.drop_last: 
            return (self.n_samples // self.batch_size)
        
        return int(np.ceil(self.n_samples / self.batch_size))