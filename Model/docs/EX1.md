```python
n_classes = 3
n_points = 400
Training_Data = SpiralData(n_points = n_points, n_classes = n_classes)
X = Training_Data.P
Y = Training_Data.L

layer1 = Dense(n_inputs = 2, n_neurons = 32)
activation1 = Activation_ReLU()
dropout_layer1 = Layer_DropOut(dropout_rate=0.05)

layer2 = Dense(n_inputs= 32, n_neurons = 32)
activation2 = Activation_ReLU()
dropout_layer2 = Layer_DropOut(dropout_rate=0.05)

layerOutput = Dense(n_inputs = 32, n_neurons = 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
accuracy_function = Accuracy_CategoricalClassification()

momentum = 0.4
learning_rate = 0.001
decay = 1e-5
epsilon = 1e-7
beta1 = 0.9
beta2 = 0.99
optimizer = Optimizer_Adam(learning_rate=learning_rate, decay_rate=decay, epsilon=epsilon, beta1=beta1, beta2=beta2)

for epoch in range(10001):
    layer1.forward(X)
    activation1.forward(layer1.output)
    dropout_layer1.forward(activation1.output)

    layer2.forward(dropout_layer1.output)
    activation2.forward(layer2.output)
    dropout_layer2.forward(activation2.output)

    layerOutput.forward(dropout_layer2.output)

    loss = loss_activation.calculate(layerOutput.output, Y)
    predictions = loss_activation.output

    if len(Training_Data.L.shape) == 2:
        Training_Data.L = np.argmax(Training_Data.L, axis = 1)

    accuracy = accuracy_function.calculate(predictions, Y)

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, LR: {optimizer.current_lr}')

    loss_activation.backward(loss_activation.output, Y)
    layerOutput.backward(loss_activation.dinputs)
    dropout_layer2.backward(layerOutput.dinputs)
    activation2.backward(dropout_layer2.dinputs)
    layer2.backward(activation2.dinputs)
    dropout_layer1.backward(layer2.dinputs)
    activation1.backward(dropout_layer2.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.update_params(layerOutput)
    optimizer.post_update_params()

TestZone = SpiralData(n_points=50, n_classes=3)
layer1.forward(TestZone.P)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layerOutput.forward(activation2.output)
predictions = np.argmax(layerOutput.output, axis=1)

plt.figure(figsize=(10, 4))

# Đồ thị 1: Dữ liệu Test thực tế
plt.subplot(1, 2, 1)
plt.title("Dữ liệu Test Thực tế")
plt.scatter(TestZone.P[:, 0], TestZone.P[:, 1], c=TestZone.L, cmap='brg', s=5)

# Đồ thị 2: Dữ liệu do Mô hình Dự đoán
plt.subplot(1, 2, 2)
plt.title("Mô hình Dự đoán")
plt.scatter(TestZone.P[:, 0], TestZone.P[:, 1], c=predictions, cmap='brg', s=5)

plt.show()
```