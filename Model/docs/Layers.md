# Layers — Code and Usage

Extracted classes for layers and regularizers from Model/model.py

## DenseLayer
```python
class DenseLayer:
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
```

## Regularizers
```python
class Regularization_L1:
    def __init__(self, strength: float = 0.0):
        self.strength = strength

    def forward(self, parameters: np.ndarray) -> float:
        return self.strength * np.sum(np.abs(parameters))

    def backward(self, parameters: np.ndarray) -> np.ndarray:
        gradient = np.ones_like(parameters)
        gradient[parameters < 0] = -1
        return self.strength * gradient

class Regularization_L2:
    def __init__(self, strength: float = 0.0):
        self.strength = strength

    def forward(self, parameters: np.ndarray) -> float:
        return self.strength * np.sum(parameters ** 2)

    def backward(self, parameters: np.ndarray) -> np.ndarray:
        return 2 * self.strength * parameters

class Regularization_L1L2:
    def __init__(self, l1_strength: float = 0.0, l2_strength: float = 0.0):
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

    def forward(self, weights: np.ndarray) -> float:
        l1_loss = self.l1_strength * np.sum(np.abs(weights))
        l2_loss = self.l2_strength * np.sum(weights ** 2)
        return l1_loss + l2_loss

    def backward(self, weights: np.ndarray) -> np.ndarray:
        l1_gradient = np.ones_like(weights)
        l1_gradient[weights < 0] = -1
        l1_gradient *= self.l1_strength
        l2_gradient = 2 * self.l2_strength * weights
        return l1_gradient + l2_gradient
```

## Layer_Dropout
```python
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
```
