import numpy as np
import matplotlib.pyplot as plt


class Dense:
    def __init__(
        self,
        n_inputs,
        n_neurons,
        seed=0
    ):
        rng = np.random.default_rng(seed)

        self.weights = (
            0.01
            * rng.standard_normal(
                (n_inputs, n_neurons)
            )
        )

        self.biases = np.zeros(
            (1, n_neurons)
        )

    def forward(self, inputs):
        self.inputs = inputs

        self.output = (
            inputs @ self.weights
            + self.biases
        )

        return self.output

    def backward(self, dvalues):
        self.dweights = (
            self.inputs.T
            @ dvalues
        )

        self.dbiases = np.sum(
            dvalues,
            axis=0,
            keepdims=True
        )

        self.dinputs = (
            dvalues
            @ self.weights.T
        )


class MeanSquaredError:
    def calculate(
        self,
        y_pred,
        y_true
    ):
        return np.mean(
            (y_pred - y_true) ** 2
        )

    def backward(
        self,
        y_pred,
        y_true
    ):
        samples = y_pred.shape[0]
        outputs = y_pred.shape[1]

        self.dinputs = (
            2
            * (y_pred - y_true)
            / (samples * outputs)
        )


class SGD:
    def __init__(
        self,
        learning_rate=0.01
    ):
        self.learning_rate = (
            learning_rate
        )

    def update_params(self, layer):
        layer.weights -= (
            self.learning_rate
            * layer.dweights
        )

        layer.biases -= (
            self.learning_rate
            * layer.dbiases
        )


np.random.seed(0)

N = 1000

X = np.linspace(
    -10,
    10,
    N
).reshape(-1, 1)

noise = (
    np.random.randn(N, 1)
    * 2.0
)

y = (
    -2.0 * X
    - 4.0
    + noise
)

dense = Dense(
    n_inputs=1,
    n_neurons=1,
    seed=0
)

loss_function = MeanSquaredError()

optimizer_instance = SGD(
    learning_rate=0.01
)

epochs = 5000

for epoch in range(epochs):
    predictions = dense.forward(X)

    loss_value = (
        loss_function.calculate(
            predictions,
            y
        )
    )

    loss_function.backward(
        predictions,
        y
    )

    dense.backward(
        loss_function.dinputs
    )

    optimizer_instance.update_params(
        dense
    )

    if epoch % 500 == 0:
        print(
            f"epoch={epoch}, "
            f"loss={loss_value:.6f}"
        )

a = dense.weights[0, 0]
b = dense.biases[0, 0]

print("Estimated equation:")
print(f"y = {a:.4f}x + {b:.4f}")

indices = np.argsort(
    X[:, 0]
)

plt.scatter(
    X[:, 0],
    y[:, 0],
    s=5,
    label="Data"
)

plt.plot(
    X[indices, 0],
    predictions[indices, 0],
    label="Prediction"
)

plt.legend()
plt.show()