import numpy as np

# 1. Cấu hình
np.random.seed(3) 
LEARNING_RATE = 0.1
EPOCHS = 10000

# 2. Dữ liệu XOR (x0: Bias, x1, x2)
x_train = [np.array([1.0, -1.0, -1.0]),
           np.array([1.0, -1.0, 1.0]),
           np.array([1.0, 1.0, -1.0]),
           np.array([1.0, 1.0, 1.0])]
y_train = [0.0, 1.0, 1.0, 0.0]

# 3. Khởi tạo trọng số (3 nơ-ron: 2 ẩn, 1 đầu ra)
# Mỗi nơ-ron nhận 3 đầu vào (1 bias + 2 input) -> 3 trọng số
n_w = [np.random.uniform(-1.0, 1.0, 3) for _ in range(3)]
n_y = [0.0, 0.0, 0.0]      # Lưu output nơ-ron
n_error = [0.0, 0.0, 0.0]  # Lưu tín hiệu lỗi (delta)

def forward_pass(x):
    # Lớp ẩn dùng tanh
    n_y[0] = np.tanh(np.dot(n_w[0], x))
    n_y[1] = np.tanh(np.dot(n_w[1], x))
    
    # Lớp đầu ra dùng sigmoid
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))

def backward_pass(y_truth):
    # Lỗi lớp đầu ra (delta_2)
    error_prime = -(y_truth - n_y[2])
    derivative = n_y[2] * (1.0 - n_y[2])
    n_error[2] = error_prime * derivative
    
    # Lỗi lớp ẩn (delta_0, delta_1)
    # Lỗi truyền ngược = (Lỗi lớp sau) * (Trọng số nối) * (Đạo hàm tanh)
    n_error[0] = n_w[2][1] * n_error[2] * (1.0 - n_y[0]**2)
    n_error[1] = n_w[2][2] * n_error[2] * (1.0 - n_y[1]**2)

def adjust_weights(x):
    # Cập nhật lớp ẩn
    n_w[0] -= LEARNING_RATE * n_error[0] * x
    n_w[1] -= LEARNING_RATE * n_error[1] * x
    
    # Cập nhật lớp đầu ra
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    n_w[2] -= LEARNING_RATE * n_error[2] * n2_inputs

# 4. Vòng lặp huấn luyện
for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(len(x_train)):
        forward_pass(x_train[i])
        backward_pass(y_train[i])
        adjust_weights(x_train[i])
        
        # Tính loss để theo dõi
        total_loss += 0.5 * (y_train[i] - n_y[2])**2
        
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss/len(x_train):.5f}")

# 5. Kiểm tra kết quả
print("\nKết quả dự đoán:")
for i in range(len(x_train)):
    forward_pass(x_train[i])
    print(f"Input: {x_train[i][1:]} | Dự đoán: {n_y[2]:.4f} | Thực tế: {y_train[i]}")