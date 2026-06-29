```python
Zone = PointZone(n_points=800, n_classes=4)

X_input = Zone.P      # Shape: (3200, 2) - Dữ liệu có 2 neuron ngõ vào
Y_true = Zone.L       # Shape: (3200,)   - Nhãn (Class từ 0 đến 3)
EPOCHS = 7001

plt.scatter(X_input[:, 0], X_input[:, 1], s=5, c=Y_true, cmap='brg')
plt.show()

dense_layer1 = Dense(n_inputs=2, n_neurons=32)
activation1 = Activation_ReLU()
dense_layer2 = Dense(n_inputs=32, n_neurons=32)
activation2 = Activation_ReLU()
layerOutput = Dense(n_inputs=32, n_neurons=4) # Output 4 class

loss_function = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.01)

for epoch in range(EPOCHS):
    dense_layer1.forward(X_input)
    activation1.forward(dense_layer1.output)
    dense_layer2.forward(activation1.output)
    activation2.forward(dense_layer2.output)
    layerOutput.forward(activation2.output)

    loss_function.calculate(layerOutput.output, Y_true)
    loss = loss_function.output

    loss_function.backward(layerOutput.output, Y_true)
    layerOutput.backward(loss_function.dinputs)
    activation2.backward(layerOutput.dinputs)
    dense_layer2.backward(activation2.dinputs)
    activation1.backward(dense_layer2.dinputs)
    dense_layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense_layer1)
    optimizer.update_params(dense_layer2)
    optimizer.update_params(layerOutput)
    optimizer.post_update_params()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

TestZone = PointZone(n_points=800, n_classes=4)
X_test = TestZone.P         # Tọa độ điểm Test (400, 2)
Y_test_true = TestZone.L    # Nhãn thực tế để đối chiếu (400,)

# 2. Quá trình DỰ ĐOÁN (Chỉ có Forward Pass, KHÔNG CÓ Backward/Optimizer)
dense_layer1.forward(X_test)
activation1.forward(dense_layer1.output)
dense_layer2.forward(activation1.output)
activation2.forward(dense_layer2.output)
layerOutput.forward(activation2.output)

# layerOutput.output lúc này có shape là (3200, 4)
# 3. Chuyển xác suất (3200, 4) thành Nhãn dự đoán (3200,) bằng argmax
predictions = np.argmax(layerOutput.output, axis=1)

plt.figure(figsize=(10, 4))

# Đồ thị 1: Dữ liệu Test thực tế
plt.subplot(1, 2, 1)
plt.title("Dữ liệu Test Thực tế")
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test_true, cmap='brg', s=5)

# Đồ thị 2: Dữ liệu do Mô hình Dự đoán
plt.subplot(1, 2, 2)
plt.title("Mô hình Dự đoán")
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='brg', s=5)

plt.show()
```