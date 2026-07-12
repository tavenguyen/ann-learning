# Mini-batch Training Guide - Hướng Dẫn Sử Dụng

## Tổng Quan

Hệ thống Mini-batch Training được thiết kế với OOP principles để cung cấp:
- **DataLoader**: Quản lý batching, shuffling, và data loading
- **MiniBatchConfig**: Cấu hình mini-batch training
- **MiniBatchTrainer**: Controller cho training loop với callbacks và validation

## 1. DataLoader - OOP Batch Loading

### Cơ Chế Hoạt Động

```python
from Model.model import DataLoader

# Tạo DataLoader
loader = DataLoader(
    X=X_train,              # Features (N, D)
    y=y_train,              # Labels (N,) hoặc (N, K)
    batch_size=32,          # Kích thước mini-batch
    shuffle=True,           # Shuffle dữ liệu
    drop_last=False,        # Có drop batch cuối cùng không
    random_state=42         # Seed cho reproducibility
)

# Iterating through batches
for X_batch, y_batch in loader:
    # X_batch: shape (batch_size, D)
    # y_batch: shape (batch_size,) hoặc (batch_size, K)
    print(f"Batch size: {len(X_batch)}")
```

### Features của DataLoader

| Feature | Mô Tả |
|---------|-------|
| **shuffle** | Shuffle dữ liệu ngẫu nhiên trước khi tạo batch |
| **drop_last** | Bỏ batch cuối nếu nhỏ hơn batch_size |
| **random_state** | Seed cho reproducibility |
| **get_batch_count()** | Lấy số batch trong epoch hiện tại |
| **get_epoch_count()** | Lấy số epoch đã lặp |
| **__len__()** | Trả về số lượng batches |

### Ví Dụ: Multi-epoch Training

```python
loader = DataLoader(X, y, batch_size=32, shuffle=True, random_state=42)

for epoch in range(3):
    print(f"Epoch {epoch}, Epoch count: {loader.get_epoch_count()}")
    for X_batch, y_batch in loader:
        # Training step
        pass
    print(f"Batches in epoch: {loader.get_batch_count()}")
```

---

## 2. MiniBatchConfig - Cấu Hình

### Khởi Tạo Config

```python
from Model.model import MiniBatchConfig

config = MiniBatchConfig(
    batch_size=32,          # Default: 32
    shuffle=True,           # Default: True
    drop_last=False,        # Default: False
    random_state=42,        # Default: None (random)
    epochs=100,             # Default: 100
    validation_split=0.2    # Chia 20% cho validation
)
```

### Tạo DataLoader từ Config

```python
train_loader = config.create_dataloader(X_train, y_train)
```

---

## 3. MiniBatchTrainer - Training Controller

### Sử Dụng Cơ Bản

```python
from Model.model import MiniBatchTrainer, MiniBatchConfig

# Setup
config = MiniBatchConfig(batch_size=32, epochs=100)
trainer = MiniBatchTrainer(model, config)

# Train
history = trainer.train(
    X=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    print_every=10,
    early_stopping_patience=10
)

# Lấy history
train_losses = history['train_loss']
val_losses = history['val_loss']
accuracies = history['train_acc']
```

### Training Loop Flow

```
For each epoch:
  1. Create DataLoader (với shuffle)
  2. For each batch:
     - Forward pass
     - Calculate loss & accuracy
     - Backward pass
     - Update parameters
  3. Validation (nếu có)
  4. Early stopping check
  5. Log results
```

### Trả Về History Dictionary

```python
history = {
    'train_loss': [...],        # Average training loss per epoch
    'train_acc': [...],         # Average training accuracy per epoch
    'train_reg_loss': [...],    # Regularization loss per epoch
    'val_loss': [...],          # Validation loss per epoch
    'val_acc': [...],           # Validation accuracy per epoch
    'lr': [...],                # Learning rate per epoch
    'batch_count': [...]        # Number of batches per epoch
}
```

---

## 4. Model.train() Integration

### Mode 1: Dùng MiniBatchTrainer (Default)

```python
# Cách 1: Đơn giản nhất
model.train(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    print_every=10,
    use_minibatch_trainer=True  # Default
)

# Cách 2: Với validation split
model.train(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,       # Chia 20% cho validation
    print_every=10
)

# Cách 3: Với early stopping
model.train(
    X=X_train,
    y=y_train,
    epochs=1000,
    batch_size=32,
    validation_data=(X_val, y_val),
    early_stopping_patience=15  # Dừng nếu val_loss không cải thiện trong 15 epochs
)
```

### Mode 2: Simple Training Loop (Compatible Mode)

```python
# Dùng code cũ (backward compatible)
model.train(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    use_minibatch_trainer=False  # Dùng simple loop
)
```

---

## 5. Advanced: MiniBatchTrainer Trực Tiếp

### Setup & Training

```python
from Model.model import MiniBatchTrainer, MiniBatchConfig

# Tạo config với các options
config = MiniBatchConfig(
    batch_size=64,
    shuffle=True,
    drop_last=False,
    random_state=42,
    epochs=200,
    validation_split=None  # Sẽ dùng validation_data
)

# Tạo trainer
trainer = MiniBatchTrainer(model, config)

# Train
history = trainer.train(
    X=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    print_every=20,
    early_stopping_patience=20
)

# Access results
print(f"Best validation loss: {trainer.best_val_loss}")
print(f"Training stopped: {trainer.early_stop}")
```

### Lấy Training History

```python
history = trainer.get_history()

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 3)
plt.plot(history['lr'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

plt.tight_layout()
plt.show()
```

---

## 6. Ví Dụ Thực Tế: Classification

```python
from Model.model import (
    Model, Layer_Dense, Activation_ReLU, 
    Activation_Softmax_Loss_CategoricalCrossEntropy,
    Optimizer_Adam, Accuracy_Categorical, DataLoader, MiniBatchTrainer
)
import numpy as np

# 1. Chuẩn bị dữ liệu
X_train = np.random.randn(1000, 10)  # 1000 samples, 10 features
y_train = np.random.randint(0, 3, 1000)  # 3 classes
X_val = np.random.randn(200, 10)
y_val = np.random.randint(0, 3, 200)

# 2. Xây dựng model
model = Model()
model.add(Layer_Dense(10, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 3))
model.add(Activation_Softmax_Loss_CategoricalCrossEntropy())

# 3. Setup model
model.set(
    loss=Activation_Softmax_Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(),
    accuracy=Accuracy_Categorical()
)
model.finalize()

# 4. Train với MiniBatchTrainer
history = model.train(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    print_every=10,
    early_stopping_patience=15,
    use_minibatch_trainer=True
)

# 5. Visualize
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.show()
```

---

## 7. Regression Example

```python
from Model.model import (
    Model, Layer_Dense, Activation_Linear, Activation_ReLU,
    Loss_MeanSquaredError, Optimizer_Adam, MiniBatchTrainer
)

# 1. Data
X_train = np.random.randn(500, 5)
y_train = np.random.randn(500, 1)  # Single output
X_val = np.random.randn(100, 5)
y_val = np.random.randn(100, 1)

# 2. Model
model = Model()
model.add(Layer_Dense(5, 32))
model.add(Activation_ReLU())
model.add(Layer_Dense(32, 16))
model.add(Activation_ReLU())
model.add(Layer_Dense(16, 1))
model.add(Activation_Linear())

# 3. Setup
model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(),
    accuracy=None  # No accuracy for regression
)
model.finalize()

# 4. Train
model.train(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    print_every=10,
    early_stopping_patience=10
)
```

---

## 8. Best Practices

### Batch Size Selection

```python
# Nhỏ (8-32): Noisier gradients, mais cuối cùng converge tốt
# Medium (32-128): Balance giữa noise và tốc độ
# Lớn (256+): Smooth gradients, tùy hardware

# Khuyến nghị:
batch_sizes = {
    'small_dataset': 32,      # < 1000 samples
    'medium_dataset': 64,     # 1k-100k samples
    'large_dataset': 128,     # > 100k samples
}
```

### Learning Rate Scheduling

```python
# Với Optimizer có decay
model.train(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    # Optimizer sẽ tự decay learning rate
)

# Kiểm tra current_lr
if hasattr(model.optimizer, 'current_lr'):
    print(f"Learning rate: {model.optimizer.current_lr}")
```

### Reproducibility

```python
# Set seed cho consistent results
np.random.seed(42)

# DataLoader với random_state
loader = DataLoader(X, y, batch_size=32, random_state=42)

# MiniBatchConfig
config = MiniBatchConfig(random_state=42)
```

### Early Stopping

```python
model.train(
    X=X_train,
    y=y_train,
    epochs=1000,  # Có thể lớn tùy ý
    batch_size=32,
    validation_data=(X_val, y_val),
    early_stopping_patience=20  # Stop nếu val_loss không cải thiện 20 epochs
)
```

---

## 9. Troubleshooting

### Problem: Validation split không work

```python
# ❌ Sai
model.train(X, y, validation_split=0.2)  # Không có batch_size

# ✅ Đúng
model.train(X, y, validation_split=0.2, batch_size=32)
```

### Problem: Memory issues với large dataset

```python
# Giảm batch size
model.train(
    X=X_train,
    y=y_train,
    batch_size=16,  # Thay vì 128
    epochs=100
)
```

### Problem: Loss không hội tụ

```python
# 1. Kiểm tra batch size
batch_size = 32  # Thử với batch size khác

# 2. Kiểm tra learning rate
# Nó có decay không?
print(model.optimizer.current_lr)

# 3. Kiểm tra validation data
# Đúng distribution với training data không?
```

---

## 10. API Reference

### DataLoader
```python
DataLoader(X, y, batch_size=32, shuffle=True, drop_last=False, random_state=None)
```
- `__iter__()`: Iterate qua batches
- `__len__()`: Số batches
- `get_batch_count()`: Batch count hiện tại
- `get_epoch_count()`: Epoch count hiện tại
- `reset()`: Reset trạng thái

### MiniBatchConfig
```python
MiniBatchConfig(batch_size=32, shuffle=True, drop_last=False, 
                random_state=None, epochs=100, validation_split=None)
```
- `create_dataloader(X, y)`: Tạo DataLoader

### MiniBatchTrainer
```python
MiniBatchTrainer(model, config=None)
```
- `train(X, y, epochs, batch_size, validation_data, ...)`: Huấn luyện
- `get_history()`: Lấy training history

### Model.train()
```python
model.train(X, y, epochs=10000, batch_size=None, print_every=100,
            validation_data=None, use_minibatch_trainer=True,
            early_stopping_patience=None, validation_split=None)
```

---

## 11. Performance Tips

1. **Prefetch batches**: DataLoader tự động shuffle hiệu quả
2. **Stateless iterations**: Mỗi epoch tạo DataLoader mới
3. **Reproducibility**: Dùng `random_state` cho consistent results
4. **Memory efficiency**: DataLoader chỉ store references, không copy data

---

## Kết Luận

Hệ thống mini-batching được thiết kế với:
- ✅ **OOP principles**: Clean, extensible, maintainable
- ✅ **Backward compatibility**: Dùng được cả cách cũ
- ✅ **Advanced features**: Early stopping, validation, history tracking
- ✅ **Flexibility**: Có thể dùng simple hoặc advanced mode
