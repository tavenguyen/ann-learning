import numpy as np
import pickle

# sử dụng Decorator
class AppRegistry:
    def __init__(self):
        # Tạo một dictionary lưu trữ ánh xạ
        self._registry = {}

    def register(self, cls):
        """Decorator để tự động đăng ký class vào hệ thống."""
        name = cls.__name__
        if name in self._registry:
            raise ValueError(f"Đã tồn tại class '{name}'")

        self._registry[name] = cls
        return cls

    def get(self, name, *args, **kwargs):
        """Khởi tạo và trả về instance của class dựa trên tên chuỗi."""
        cls = self._registry.get(name)
        if not cls:
            raise KeyError(f"Không tìm thấy class '{name}' trong Registry!")
        # Khởi tạo instance với các tham số truyền vào (nếu có)
        return cls(*args, **kwargs)
    
    def list_classes(self):
        return list(self._registry.keys())

CLASS_REGISTRY = AppRegistry()

#----------------------------- DataPoints -------------------------------#
class Line:
    def __init__(self, n_points, n_classes, n_dimensions, noise_std=2.0, random_state=None):
        # Use local RNG for reproducibility
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
            a = 2 * (j - 1)
            b0 = (j - 2) * 2
            ix = range(N * j, N * (j + 1))
            t = np.linspace(-10, 10, N)

            noise = rng.normal(loc=0.0, scale=noise_std, size=N)

            if D == 2:
                y = a * t + b0 + noise
                self.P[ix] = np.c_[t, y]
            else:
                raise NotImplementedError("Only D==2 supported for Line data generator.")
            self.L[ix] = j

class Quadratic:
    def __init__(self, n_points, a, b, c, x_min = -10.0, x_max = 10.0, noise_std = 2.0):
        self.a = a
        self.b = b
        self.c = c

        self.X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)

        # noise ~ N(0, noise_std^2)
        noise = np.random.randn(n_points, 1) * noise_std
        self.Y = (a * self.X ** 2 + b * self.X + c) + noise

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

class Circle2D:
    def __init__(self, n_points, n_classes, n_dimensions, noise_std=2.0, radius_step=2.0, random_state=None):
        """Generate concentric noisy circles (2D only).

        Parameters
        - n_points: points per class
        - n_classes: number of concentric circles
        - n_dimensions: must be 2
        - noise_std: std dev of radial noise
        - radius_step: base step between radii (radius = radius_step * (j+1))
        - random_state: optional int for reproducibility
        """
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()

        N = n_points
        K = n_classes
        D = n_dimensions

        if D != 2:
            raise NotImplementedError("Circle2D supports only 2D (n_dimensions==2)")

        self.P = np.zeros((N * K, D))
        self.L = np.zeros(N * K, dtype='uint8')

        for j in range(K):
            base_r = radius_step * (j + 1)
            r_noise = base_r + rng.normal(scale=noise_std, size=N)

            theta = np.linspace(0, 2 * np.pi, N)
            idx = range(N * j, N * (j + 1))

            # polar -> Cartesian
            self.P[idx] = np.c_[r_noise * np.cos(theta), r_noise * np.sin(theta)]
            self.L[idx] = j

#----------------------------- Dense -------------------------------#

@CLASS_REGISTRY.register
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
        
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
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
            "n_inputs": self.nInputs,
            "n_neurons": self.nNeurons,
            "weight_regularizer": self.weight_regularizer.getConfig() if self.weight_regularizer else None,
            "bias_regularizer": self.bias_regularizer.getConfig() if self.bias_regularizer else None
        }
     
    def getParameters(self):
        return {
            "weights": self.weights,
            "biases": self.biases
        }
    
    def setParameters(self, parameters):
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]

@CLASS_REGISTRY.register
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
    
    def getConfig(self):
        return {
            "className": "Regularization_L1",
            "strength": self.strength
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, parameters):
        return None
    
@CLASS_REGISTRY.register
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
    
    def getConfig(self):
        return {
            "className": "Regularization_L2",
            "strength": self.strength
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, parameters):
        return None

@CLASS_REGISTRY.register
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

    def getConfig(self):
        return {
            "className": "Regularization_L1L2",
            "l1_strength": self.l1_strength,
            "l2_strength": self.l2_strength
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, parameters):
        return None

# chi su dung trong training
@CLASS_REGISTRY.register
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
    
    def getConfig(self):
        return {
            "className": "Layer_Dropout",
            "dropout_rate": self.dropout_rate
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, parameters):
        return None

#----------------------------- Activation -------------------------------#
@CLASS_REGISTRY.register
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
            "className": "Activation_Linear"
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass
    
@CLASS_REGISTRY.register
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
            "className": "Activation_ReLU"
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values / (np.sum(exp_values, axis = 1, keepdims=True))
        
        self.output = probabilities
        
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output,single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

    def getConfig(self):
        return {
            "className": "Activation_Softmax"
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
class Activation_Sigmoid:
    def forward(self, inputs):
        # Save inputs and compute sigmoid activation
        self.inputs = inputs
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):
        # Derivative of sigmoid: s * (1 - s)
        self.dinputs = dvalues * (self.output * (1.0 - self.output))
        return self.dinputs

    def getConfig(self):
        return {
            "className": "Activation_Sigmoid"
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
    
@CLASS_REGISTRY.register
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
            "className": "Activation_Softmax_Loss_CategoricalCrossEntropy"
        }
    
    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
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
            "className": "Loss_MeanSquaredError"
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
@CLASS_REGISTRY.register
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
            "className": "Accuracy_RegressionTolerance",
            "tolerance": self.tolerance
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
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
            "className": "Accuracy_CategoricalClassification"
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

#----------------------------- Optimizer-------------------------------#
@CLASS_REGISTRY.register
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
            "className": "Optimizer_SGD",
            "learning_rate": self.learning_rate
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
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
            "className": "Optimizer_SGD_Decay",
            "learning_rate": self.initial_lr,
            "decay_rate": self.decay_rate
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
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
            "className": "Optimizer_SGD_Momentum",
            "learning_rate": self.learning_rate,
            "beta": self.beta
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
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
            "className": "Optimizer_Momentum_Decay",
            "learning_rate": self.initial_lr,
            "decay_rate": self.decay_rate,
            "beta": self.beta
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
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
            "className": "Optimizer_AdaGrad",
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
class Optimizer_RMSProp:
    def __init__(self, learning_rate = 0.001, p = 0.9, epsilon = 1e-7):
        self.initial_lr = learning_rate
        self.current_lr = learning_rate
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

        layer.weights -= self.current_lr * layer.dweights / (np.sqrt(layer.moving_average_weight) + self.epsilon)
        layer.biases -= self.current_lr * layer.dbiases / (np.sqrt(layer.moving_average_bias) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
        
    def getConfig(self):
        return {
            "className": "Optimizer_RMSProp",
            "learning_rate": self.initial_lr,
            "p": self.p,
            "epsilon": self.epsilon
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

@CLASS_REGISTRY.register
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

    def getConfig(self):
        return {
            "className": "Optimizer_Adam",
            "learning_rate": self.initial_lr,
            "decay_rate": self.decay_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        }

    def getParameters(self):
        return None
    
    def setParameters(self, paramemters):
        pass

#----------------------------- Model -------------------------------#
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
    def train(self, X, y, epochs=10000, batch_size=None, print_every=100, validation_data=None):
        """Huấn luyện mô hình với Mini-batch Gradient Descent.
        
        Parameters
        ----------
        X : np.ndarray
            Dữ liệu đầu vào (Features).
        y : np.ndarray
            Nhãn mục tiêu (Labels).
        epochs : int
            Số vòng lặp huấn luyện.
        batch_size : int, optional
            Kích thước batch. Nếu None, huấn luyện theo Full-batch.
        print_every : int
            Tần suất in log ra console.
        validation_data : tuple (X_val, y_val), optional
            Dữ liệu để kiểm thử sau mỗi epoch báo cáo.
        """
        # Đảm bảo model đã được finalize trước khi train
        if not hasattr(self, "trainable_layers"):
            self.finalize()

        if self.accuracy is not None:
            self.accuracy.init(y)

        # Nếu không truyền batch_size, ta sử dụng toàn bộ dữ liệu (Full-batch)
        bs = batch_size if batch_size is not None else len(X)

        for epoch in range(epochs + 1):
            # Khởi tạo lại trạng thái Accuracy cho epoch mới
            if self.accuracy is not None:
                self.accuracy.new_pass()
            
            epoch_data_loss = 0
            epoch_reg_loss = 0
            
            # Khởi tạo DataLoader cho epoch này (để shuffle dữ liệu)
            train_loader = DataLoader(X, y, batch_size=bs, shuffle=True)
            
            for X_batch, y_batch in train_loader:
                # 1. Forward pass
                y_pred = self.forward(X_batch, training=True)

                # 2. Tính Loss
                data_loss = self.loss.calculate(y_pred, y_batch)
                reg_loss = self.get_regularization_loss()
                
                # 3. Tính Accuracy (hàm calculate sẽ tự tích lũy vào instance accuracy)
                if self.accuracy is not None:
                    self.accuracy.calculate(y_pred, y_batch)

                # 4. Backward pass
                self.backward(y_pred, y_batch)

                # 5. Cập nhật trọng số
                self.update_params()
                
                epoch_data_loss += data_loss
                epoch_reg_loss += reg_loss

            # Tính toán các chỉ số trung bình sau một epoch
            n_batches = len(train_loader)
            avg_data_loss = epoch_data_loss / n_batches
            avg_reg_loss = epoch_reg_loss / n_batches
            avg_loss = avg_data_loss + avg_reg_loss
            avg_acc = self.accuracy.calculate_accumulated() if self.accuracy is not None else None

            # Báo cáo kết quả
            if epoch % print_every == 0:
                stats = f"epoch: {epoch}, loss: {avg_loss:.5f} (data: {avg_data_loss:.5f}, reg: {avg_reg_loss:.5f})"
                if avg_acc is not None:
                    stats += f", acc: {avg_acc:.5f}"
                
                # Kiểm tra kết quả trên tập Validation (nếu có)
                if validation_data:
                    val_loss, val_acc = self.evaluate(*validation_data)
                    stats += f" | val_loss: {val_loss:.5f}"
                    if val_acc is not None:
                        stats += f", val_acc: {val_acc:.5f}"
                
                # Hiển thị learning rate hiện tại (nếu optimizer có decay)
                if hasattr(self.optimizer, 'current_lr'):
                    stats += f", lr: {self.optimizer.current_lr:.6f}"
                
                print(stats)

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

    @classmethod
    def createobjectfromconfig(cls, config):
        if config is None: return None
        
        class_name = config.get("className")
        # Lấy các tham số còn lại, loại bỏ className
        kwargs = {k: v for k, v in config.items() if k != "className"}
        
        # Xử lý đệ quy cho các đối tượng lồng (ví dụ: regularizers)
        for k, v in kwargs.items():
            if isinstance(v, dict) and "className" in v:
                kwargs[k] = cls.createobjectfromconfig(v)
                
        return CLASS_REGISTRY.get(class_name, **kwargs)
        
    @classmethod
    def createFromConfig(cls, config):
        model = cls()
        for layerConfig in config["layers"]:
            layer = cls.createobjectfromconfig(layerConfig)
            model.add(layer)

        lossConfig = config["loss"]

        if lossConfig is not None:
            loss = cls.createobjectfromconfig(
                lossConfig
            )
        else:
            loss = None

        optimizerConfig = config.get(
            "optimizer"
        )

        if optimizerConfig is not None:
            optimizer = cls.createobjectfromconfig(
                optimizerConfig
            )
        else:
            optimizer = None

        accuracyConfig = config.get(
            "accuracy"
        )

        if accuracyConfig is not None:
            accuracy = cls.createobjectfromconfig(
                accuracyConfig
            )
        else:
            accuracy = None

        model.set(
            loss=loss,
            optimizer=optimizer,
            accuracy=accuracy
        )

        model.finalize()

        return model

    def getParameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.getParameters())

        return parameters

    # Phải gọi finalize() trước khi setParameters() vì finalize() tạo self.trainable_layers.
    def setParameters(self, parameters):
        if len(parameters) != len(self.trainable_layers):
            raise ValueError(
                "Số lượng parameter không khớp với số trainable layers."
            )

        for layer, layerParameters in zip(
            self.trainable_layers,
            parameters
        ):
            layer.setParameters(
                layerParameters
            )

    def getOptimizerState(self):
        """Thu thập trạng thái nội tại của optimizer (velocity, caches) từ các layer."""
        state = []
        for layer in self.trainable_layers:
            layer_state = {}
            # Các thuộc tính trạng thái phổ biến mà optimizer gắn vào layer
            attrs = ["weight_velocity", "bias_velocity", 
                     "weight_cache", "bias_cache",
                     "weight_momentum", "bias_momentum",
                     "moving_average_weight", "moving_average_bias"]
            
            for attr in attrs:
                if hasattr(layer, attr):
                    layer_state[attr] = getattr(layer, attr)
            
            state.append(layer_state if layer_state else None)
        
        return {
            "iterations": self.optimizer.iterations,
            "layer_states": state
        }

    def setOptimizerState(self, state):
        """Khôi phục trạng thái optimizer để tiếp tục huấn luyện."""
        if state is None: return
        
        self.optimizer.iterations = state["iterations"]
        for layer, layer_state in zip(self.trainable_layers, state["layer_states"]):
            if layer_state:
                for key, value in layer_state.items():
                    setattr(layer, key, value)

    def save(self, filepath):
        """Lưu toàn bộ mô hình (config, weights, optimizer state) vào file."""
        data = {
            "config": self.getConfig(),
            "parameters": self.getParameters(),
            "optimizerState": self.getOptimizerState()
        }

        with open(filepath, "wb") as file:
            pickle.dump(data, file)
        print(f"Đã lưu mô hình thành công vào file: '{filepath}'")

    @classmethod
    def load(cls, filepath):
        """Khôi phục mô hình hoàn chỉnh từ file."""
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        model = cls.createFromConfig(data["config"])
        model.setParameters(data["parameters"])
        model.setOptimizerState(data.get("optimizerState"))
        
        print(f"Đã tải mô hình thành công từ file: '{filepath}'")
        return model

#----------------------------- Mini-batch Training -------------------------------#
class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True, drop_last=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
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
            
            # cú pháp slicing sẽ tự dừng ở phần tử cuối cùng nếu end lớn hơn n_samples
            x_batches = X_shuffled[start:end]
            y_batches = y_shuffled[start:end]
            yield x_batches, y_batches

    def __len__(self):
        if self.drop_last: 
            return (self.n_samples // self.batch_size)
        
        return int(np.ceil(self.n_samples / self.batch_size))