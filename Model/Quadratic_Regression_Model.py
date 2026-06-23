import model as Model
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

inputs = Quadratic(n_points = 10, a = 2.0, b = 3.0, c = 5.0, noise_std=2.0)
X = inputs.X
Y = inputs.Y
plt.scatter(X, Y, s = 5)
plt.show()

X_inputs = np.c_[X ** 2, X]

quadratic_model = Model.Model()
quadratic_model.add(Model.DenseLayer(n_inputs = 2, n_neurons = 1))
quadratic_model.add(Model.Activation_Linear())
quadratic_model.set(loss = Model.Loss_MeanSquaredError(), optimizer = Model.Optimizer_Adam(learning_rate=0.001), accuracy = None)
quadratic_model.finalize()
quadratic_model.train(X_inputs, Y)

dense = quadratic_model.layers[0]
A = dense.weights[0, 0]
B = dense.weights[1, 0]
C = dense.biases[0, 0]
print("Learned parameters:")
print("a =", A)
print("b =", B)
print("c =", C)

Y_pred = A * X ** 2 + B * X + C

plt.scatter(X, Y, s = 5)

plt.plot(
    X,
    Y_pred,
    label="Prediction"
)
plt.show()