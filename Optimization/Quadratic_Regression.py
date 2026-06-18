import numpy as np
import matplotlib.pyplot as plt

class Quadratic:
    def __init__(self, n_points, a, b, c, x_min = -10.0, x_max = 10.0, noise_std = 2.0):
        self.a = a
        self.b = b
        self.c = c

        self.X = np.linspace(x_min, x_max, n_points).reshape(-1, 1)

        # noise ~ N(0, noise_std^2)
        noise = np.random.randn(n_points, 1) * noise_std
        self.Y = (a * self.X ** 2 + b * self.X + c) + noise

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

class Optimizer_SGD:
    def __init__(self, lr = 0.001):
        self.learning_rate = lr
        self.iterations = 0

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    
    # Đạo hàm của activation linear = 1 nên lấy dvalues vào
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

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
        self.output = np.mean(samples)
        return self.output
    
class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        loss = np.mean(squared_error, axis = 1)
        return loss

    # error = 1/N \sum_M (y_pred - y_true)^2
    def backward(self, y_pred, y_true):
        N = y_pred.shape[0]
        M = y_pred.shape[1]

        self.dinputs = 2 * (y_pred - y_true) / (N * M)
        return self.dinputs

class DenseLayer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    
N = 10
np.random.seed(0)

epochs = 20000
learning_rate = 0.001

inputs = Quadratic(n_points=N, a = 2.0, b = -3.0, c = 4.0, noise_std=2.0, x_min=-5.0, x_max=5.0)
X = inputs.X
Y = inputs.Y

plt.scatter(X, Y, s = 5)
plt.show()

layer1 = DenseLayer(n_inputs=2, n_neurons=1)
activation1 = Activation_Linear()
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_Adam(learning_rate = learning_rate)

X_input = np.zeros((N, 2))
for i in range(N):
    X_input[i, 0] = X[i] ** 2
    X_input[i, 1] = X[i]

for epoch in range(epochs):
    layer1.forward(X_input)
    activation1.forward(layer1.output)

    loss = loss_function.calculate(activation1.output, Y)

    y_pred = activation1.output
    mae = np.mean(
        np.abs(y_pred - Y)
    )

    rmse = np.sqrt(
        np.mean(
            (y_pred - Y) ** 2
        )
    )

    ss_res = np.sum(
        (Y - y_pred) ** 2
    )

    ss_total = np.sum(
        (Y - np.mean(Y)) ** 2
    )

    r2 = 1 - ss_res / ss_total

    # backward pass
    loss_function.backward(activation1.output, Y)
    activation1.backward(loss_function.dinputs)
    layer1.backward(activation1.dinputs)

    # optimizer
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.post_update_params()

    if epoch % 500 == 0:
        print("Loss: ", loss, " MAE: ", mae, " RMSE", rmse, " R^2", r2)

a = layer1.weights[0, 0]
b = layer1.weights[1, 0]
c = layer1.biases[0, 0]

Y_pred = a * X ** 2 + b * X + c

print("Learned parameters:")
print("a =", a)
print("b =", b)
print("c =", c)

plt.scatter(X, Y, s = 5)

plt.plot(
    X,
    Y_pred,
    label="Prediction"
)

plt.legend()
plt.show()