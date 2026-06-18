import numpy as np
import matplotlib.pyplot as plt

class Line:
    def __init__(self, n_points, n_classes, n_dimensions):
        N = n_points
        K = n_classes
        D = n_dimensions

        self.P = np.zeros((N * K, D))
        self.L = np.zeros(N * K, dtype = 'uint8')
        # y = ax + b
        for j in range(K):
            a = 2 * (j - 1)
            b = (j - 2) * 2 + np.random.randn(N) * 2  
            ix = range(N * j, N * (j + 1))
            t = np.linspace(-10, 10, N)
            if D == 2:
                self.P[ix] = np.c_[t, a * t + b]

            self.L[ix] = j 

class Activation_ReLU:
    def forward(self, Z):
        self.inputs = Z
        self.output =  np.maximum(0, Z)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
class Activation_Linear:
    def forward(self, Z):
        self.inputs = Z
        self.output = Z

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Optimizer_SGD:
    def __init__(self, lr = 1.0):
        self.learning_rate = lr
        self.iterations = 0

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases
    
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(
        self,
        learning_rate=0.001,
        decay_rate=0.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    ):
        self.initial_learning_rate = (
            learning_rate
        )

        self.current_learning_rate = (
            learning_rate
        )

        self.decay_rate = decay_rate

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.epsilon = epsilon

        self.iterations = 0

    def pre_update_params(self):
        self.current_learning_rate = (
            self.initial_learning_rate
            / (
                1
                + self.decay_rate
                * self.iterations
            )
        )

    def update_params(self, layer):
        if not hasattr(
            layer,
            "weight_momentums"
        ):
            layer.weight_momentums = (
                np.zeros_like(
                    layer.weights
                )
            )

            layer.bias_momentums = (
                np.zeros_like(
                    layer.biases
                )
            )

            layer.weight_cache = (
                np.zeros_like(
                    layer.weights
                )
            )

            layer.bias_cache = (
                np.zeros_like(
                    layer.biases
                )
            )

        layer.weight_momentums = (
            self.beta_1
            * layer.weight_momentums
            + (1 - self.beta_1)
            * layer.dweights
        )

        layer.bias_momentums = (
            self.beta_1
            * layer.bias_momentums
            + (1 - self.beta_1)
            * layer.dbiases
        )

        layer.weight_cache = (
            self.beta_2
            * layer.weight_cache
            + (1 - self.beta_2)
            * layer.dweights ** 2
        )

        layer.bias_cache = (
            self.beta_2
            * layer.bias_cache
            + (1 - self.beta_2)
            * layer.dbiases ** 2
        )

        weight_momentums_corrected = (
            layer.weight_momentums
            / (
                1
                - self.beta_1
                ** (self.iterations + 1)
            )
        )

        bias_momentums_corrected = (
            layer.bias_momentums
            / (
                1
                - self.beta_1
                ** (self.iterations + 1)
            )
        )

        weight_cache_corrected = (
            layer.weight_cache
            / (
                1
                - self.beta_2
                ** (self.iterations + 1)
            )
        )

        bias_cache_corrected = (
            layer.bias_cache
            / (
                1
                - self.beta_2
                ** (self.iterations + 1)
            )
        )

        layer.weights -= (
            self.current_learning_rate
            * weight_momentums_corrected
            / (
                np.sqrt(
                    weight_cache_corrected
                )
                + self.epsilon
            )
        )

        layer.biases -= (
            self.current_learning_rate
            * bias_momentums_corrected
            / (
                np.sqrt(
                    bias_cache_corrected
                )
                + self.epsilon
            )
        )

    def post_update_params(self):
        self.iterations += 1

class Loss:
    def calculate(self, y_pred, y_true):
        samples = self.forward(y_pred, y_true)
        self.loss = np.mean(samples)
        return self.loss
    
class Loss_MeanSquared_Error(Loss):
    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        self.output = np.mean(squared_error, axis = 1)
        return self.output
    
    def backward(self, y_pred, y_true):
        samples = y_pred.shape[0] 
        outputs = y_pred.shape[1]
        # y_pred, y_true co shape (N, M) voi N la so luong mau, M la so output trong moi mau

        self.dinputs = (2 * (y_pred - y_true)) / (outputs * samples)
        return self.dinputs

class DenseLayer:
    def __init__(self, n_neurons, n_inputs, weight_regularizer = None, bias_regularizer = None):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases    
        return self.output
    
    def backward(self, dvalues):
        #dZ = dA * self.activation.backward(self.output)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True)
        dX = np.dot(dvalues, self.weights.T)
        return dX

np.random.seed(0)    

N = 1000
K = 1
D = 2
learning_rate = 0.01
EPOCHS = 10000

inputs = Line(n_points = N, n_classes= K, n_dimensions = D)

X = np.zeros((N, 1))
Y = np.zeros((N, 1))
for i in range(N):
    X[i, 0] = inputs.P[i, 0]
    Y[i, 0] = inputs.P[i, 1]
    
plt.scatter(X, Y, c = inputs.L, s = 5)
plt.show()

layer = DenseLayer(n_inputs=1, n_neurons=1)
activation1 = Activation_Linear()
loss_function = Loss_MeanSquared_Error()
optimizer = Optimizer_Adam(learning_rate=learning_rate)

for epoch in range(EPOCHS):
    layer.forward(X)
    activation1.forward(layer.output)

    loss = loss_function.calculate(activation1.output, Y)
    mae = np.mean(np.abs(activation1.output - Y))

    #backward
    loss_function.backward(activation1.output, Y)
    activation1.backward(loss_function.dinputs)
    layer.backward(activation1.dinputs)

    # update_params
    optimizer.pre_update_params()
    optimizer.update_params(layer)
    optimizer.post_update_params()

    if epoch % 500 == 0:
        print(loss)
        print('Mean Absolute Error: ', mae)

a = layer.weights[0, 0]
b = layer.biases[0, 0]
Y_pred = a * X + b

print("Learned parameters:")
print("a =", a)
print("b =", b)

plt.scatter(X, Y, c = inputs.L, s = 40)

plt.plot(
    X,
    Y_pred,
    label="Prediction"
)

plt.legend()
plt.show()