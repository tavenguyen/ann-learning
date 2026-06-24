# Code Review Chuyên sâu — Model/model.py (ANN Implementation)

## Executive Summary
File model.py triển khai **Feedforward Neural Network (FNN)** với **backpropagation** và nhiều optimizer (SGD, Momentum, Adam, RMSProp). Mã hoạt động cơ bản nhưng thiếu:
1. **Model persistence** (Save/Load/Create) → mỗi lần chạy lại phải retrain
2. **Numerical stability** (incorrect Softmax+CCE, no gradient clipping)
3. **API consistency** (Loss backward signature, Accuracy state management)
4. **Training robustness** (no mini-batch, no early stopping, no validation protocol)

---

## PHẦN 1: LỖI CRITICAL (TIER 1) — PHẢI SỬA NGAY

### 1. Model Persistence — COMPLETELY MISSING (Ưu tiên 1.1)

**Vấn đề:**
- Không có `model.save()` → train xong weights biến mất khi process thoát.
- Không có `model.load()` → không thể nạp mô hình đã save.
- Không có `Model.create_from_config()` → không thể định nghĩa architecture qua JSON/dict.
- Mỗi lần chạy script → retrain từ đầu (낭비thời gian compute).

**Hậu quả:**
- **No reproducibility:** Không thể share trained models.
- **No deployment:** Mô hình không thể lưu cho production inference.
- **Loss of trained parameters:** Tất cả learning bị mất sau thoát.
- **No checkpointing:** Không lưu best model during validation.

**Thuật ngữ chuyên ngành:**
- **Serialization:** Chuyển đổi object (weights, architecture) thành bytes/file format.
- **Deserialization:** Reconstruct object từ bytes/file.
- **Checkpoint:** Lưu state của model tại iteration/epoch nhất định (để recover best model).
- **Model architecture:** Danh sách layers, kích thước, activation functions.
- **Model weights (parameters):** Giá trị thực tế của w, b sau training.

**Recommended Implementation:**

```python
import pickle
import json
import numpy as np

class Model:
    # ... existing code ...
    
    def save(self, filepath: str, include_optimizer: bool = False) -> None:
        """Serialize trained model weights & architecture.
        
        Parameters
        ----------
        filepath : str
            Output path (*.pkl for pickle, *.npz for numpy format)
        include_optimizer : bool
            Whether to save optimizer state (untuk resume training)
            
        Notes
        -----
        Saves only weights/biases, not activation functions (can be recreated).
        Use .pkl for full serialization or .npz for lightweight weight-only save.
        """
        model_data = {
            'layers_info': [],
            'optimizer_config': None
        }
        
        # Save layer weights & architecture
        for layer in self.layers:
            layer_info = {
                'type': layer.__class__.__name__,
                'weights': layer.weights.copy() if hasattr(layer, 'weights') else None,
                'biases': layer.biases.copy() if hasattr(layer, 'biases') else None,
            }
            
            # Save regularizer config
            if hasattr(layer, 'weight_regularizer') and layer.weight_regularizer:
                layer_info['weight_regularizer'] = {
                    'type': layer.weight_regularizer.__class__.__name__,
                    'strength': getattr(layer.weight_regularizer, 'strength', None)
                }
            
            model_data['layers_info'].append(layer_info)
        
        # Optional: save optimizer state (untuk warm-start training)
        if include_optimizer and hasattr(self, 'optimizer'):
            model_data['optimizer_config'] = {
                'type': self.optimizer.__class__.__name__,
                'iterations': getattr(self.optimizer, 'iterations', 0),
                'learning_rate': getattr(self.optimizer, 'learning_rate', None),
                'current_lr': getattr(self.optimizer, 'current_lr', None),
            }
        
        # Serialize thành file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'Model':
        """Deserialize trained model từ file.
        
        Parameters
        ----------
        filepath : str
            Path to saved model (*.pkl)
            
        Returns
        -------
        Model
            Loaded model với weights & biases restored.
            
        Notes
        -----
        Activation functions & loss không được save; cần set() lại sau load.
        Ví dụ:
            model = Model.load('model.pkl')
            model.set(loss=Activation_Softmax_Loss_CategoricalCrossEntropy(),
                     optimizer=Optimizer_Adam())
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct model
        model = Model()
        
        # Tạo layers mới & restore weights
        for layer_info in model_data['layers_info']:
            layer_type = layer_info['type']
            
            if layer_type == 'DenseLayer':
                # Suy ra n_inputs, n_neurons từ weights shape
                n_inputs, n_neurons = layer_info['weights'].shape
                layer = DenseLayer(n_neurons, n_inputs)
                layer.weights = layer_info['weights'].copy()
                layer.biases = layer_info['biases'].copy()
                
                # Restore regularizer if saved
                if 'weight_regularizer' in layer_info and layer_info['weight_regularizer']:
                    reg_type = layer_info['weight_regularizer']['type']
                    strength = layer_info['weight_regularizer']['strength']
                    if reg_type == 'Regularization_L1':
                        layer.weight_regularizer = Regularization_L1(strength)
                    elif reg_type == 'Regularization_L2':
                        layer.weight_regularizer = Regularization_L2(strength)
                
            elif layer_type == 'Activation_ReLU':
                layer = Activation_ReLU()
            elif layer_type == 'Activation_Linear':
                layer = Activation_Linear()
            # ... thêm các activation khác nếu cần ...
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            model.add(layer)
        
        model.finalize()
        print(f"Model loaded from {filepath}")
        return model
    
    def save_weights_only(self, filepath: str) -> None:
        """Lưu chỉ weights (nhẹ hơn) trong format NumPy .npz.
        
        Notes
        -----
        Dùng khi chỉ cần weights, không cần architecture metadata.
        """
        weights_dict = {}
        for i, layer in enumerate(self.trainable_layers):
            weights_dict[f'layer_{i}_weights'] = layer.weights
            weights_dict[f'layer_{i}_biases'] = layer.biases
        
        np.savez(filepath, **weights_dict)
        print(f"Weights saved to {filepath}")
    
    @staticmethod
    def load_weights_only(filepath: str, model: 'Model') -> None:
        """Restore weights từ .npz file vào existing model.
        
        Parameters
        ----------
        filepath : str
            Path to .npz file
        model : Model
            Model object để inject weights vào
        """
        data = np.load(filepath)
        for i, layer in enumerate(model.trainable_layers):
            layer.weights = data[f'layer_{i}_weights']
            layer.biases = data[f'layer_{i}_biases']
        print(f"Weights loaded from {filepath}")


# Sử dụng ví dụ:
# model.save('trained_model.pkl')
# loaded = Model.load('trained_model.pkl')
# loaded.set(loss=loss_fn, optimizer=optimizer)
# predictions = loaded.predict(X_test)
```

---

### 2. Softmax + Categorical Cross-Entropy — Numerical Bugs (Ưu tiên 1.2)

**Vấn đề:**
- ✅ **Đã sửa ở trên:** Clip áp lên probabilities, indexing đúng, one-hot handling.
- **Còn lại:** backward signature đúng `backward(self, y_pred, y_true)` & sử dụng `self.output` (stored probs).
- **Impact:** Gradient flow chính xác → training hội tụ.

**Thuật ngữ:**
- **Softmax:** σ(z_i) = e^(z_i) / Σ_j e^(z_j) → normalizationthành probability distribution.
- **Categorical Cross-Entropy:** L = -Σ y_true * log(y_pred) → measure divergence hai distributions.
- **Log-sum-exp trick:** Tính exp(z - max(z)) để tránh overflow.
- **Gradient:** ∂L/∂z = softmax(z) - y_true (closed form cho Softmax+CCE).

---

### 3. Accuracy State Initialization & Inheritance (Ưu tiên 1.3)

**Vấn đề:**
- ✅ **Đã sửa:** Accuracy_CategoricalClassification kế thừa Accuracy.
- **Nhưng:** Cần đảm bảo `new_pass()` được gọi trước mỗi epoch.
- **Risk:** Silent bug nếu không gọi new_pass() → accumulated metrics sai.

**Thuật ngữ:**
- **Accumulation:** Tích lũy metric qua multiple mini-batches để tính epoch-level accuracy.
- **State machine:** new_pass() → [batch calculate] × N → calculate_accumulated() → reset.

---

## PHẦN 2: LỖI HIGH (TIER 2) — NÊN SỬA SỚM

### 4. Weight Initialization — No Xavier/He (Ưu tiên 2.1)

**Vấn đề:**
```python
# Current: Generic N(0,1) initialization
self.weights = np.random.randn(n_inputs, n_neurons)

# Problem: Variance không phù hợp với activation function
# → Vanishing gradient (sigmoid/tanh) hoặc Exploding gradient (ReLU)
```

**Cải thiện:**
```python
def __init__(self, n_neurons: int, n_inputs: int, weight_init='he'):
    if weight_init == 'he':  # Cho ReLU
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
    elif weight_init == 'xavier':  # Cho sigmoid/tanh (cân bằng)
        limit = np.sqrt(6.0 / (n_inputs + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
    else:
        self.weights = np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
```

**Thuật ngữ:**
- **He initialization:** w ~ N(0, √(2/n_in)) → giữ variance gradient = 1 qua ReLU layers.
- **Xavier/Glorot:** w ~ U[-√(6/(n_in+n_out)), √(6/(n_in+n_out))] → cân bằng forward/backward.
- **Vanishing gradient:** ∂L/∂w → 0 khi sâu → training chậm.
- **Exploding gradient:** ∂L/∂w → ∞ (NaN) → training không ổn định.

---

### 5. Optimizer State Inconsistency (Ưu tiên 2.2)

**Vấn đề:**
```python
# Tên biến khác nhau giữa optimizer
Optimizer_SGD: learning_rate
Optimizer_SGD_Decay: current_lr, initial_lr
Optimizer_Adam: current_lr

# pre_update_params() nhiều chỗ vô nghĩa
def pre_update_params(self):
    self.learning_rate = self.learning_rate  # ??? vô nghĩa
```

**Cải thiện:** Thống nhất `current_lr` & decay calculation trong pre_update.

**Thuật ngữ:**
- **Learning rate decay:** lr(t) = lr₀ / (1 + decay_rate * t) → giảm lr theo epoch.
- **Exponential decay:** lr(t) = lr₀ * γ^t (ví dụ γ=0.95).
- **Step decay:** lr(t) = lr₀ * α^⌊t/step⌋ → giảm theo từng bước cố định.

---

### 6. Mini-batch & Data Shuffling (Ưu tiên 2.3)

**Vấn đề:**
```python
# Current: Full-batch GD
def train(self, x, y, epochs=10000, ...):
    for epoch in range(epochs):
        y_pred = self.forward(x)  # ← toàn bộ dataset 1 lần
```

**Cải thiện:**
```python
def train(self, x, y, epochs=10000, batch_size=32, shuffle=True, ...):
    n_samples = len(x)
    for epoch in range(epochs):
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)  # Tránh ordering bias
        
        epoch_loss, epoch_acc = 0, 0
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch, y_batch = x[batch_indices], y[batch_indices]
            
            y_pred = self.forward(x_batch)
            loss = self.loss.calculate(y_pred, y_batch)
            # ... backward, update ...
            epoch_loss += loss * len(batch_indices)
        
        epoch_loss /= n_samples
        # print & validate
```

**Thuật ngữ:**
- **Batch size:** Số samples per update (trade-off: noise ↔ efficiency).
- **Mini-batch GD:** Update sau mỗi batch (trade-off SGD & full-batch).
- **Shuffling:** Random reorder samples → tránh ordering bias.
- **Epoch:** Một lần đi qua toàn bộ training set.

---

## PHẦN 3: LỖI MEDIUM (TIER 3) — NÂNG CAO CHẤT LƯỢNG

### 7. Gradient Clipping (Ưu tiên 3.1)

**Vấn đề:** Khi ||∇|| quá lớn → exploding gradient → NaN loss.

**Cải thiện:**
```python
def update_params(self):
    self.optimizer.pre_update_params()
    
    for layer in self.trainable_layers:
        # Gradient clipping trước update
        if hasattr(layer, 'dweights'):
            layer.dweights = np.clip(layer.dweights, -1, 1)  # L∞ norm
        self.optimizer.update_params(layer)
    
    self.optimizer.post_update_params()
```

**Thuật ngữ:**
- **Gradient clipping:** ||∇|| > max_norm → scale to max_norm.
- **L2 norm clipping:** ||∇||₂ → ||∇||₂ / max(1, ||∇||₂/max_norm).

---

### 8. Early Stopping & Validation Protocol (Ưu tiên 3.2)

**Vấn đề:** Không biết khi nào dừng training → overfitting lâu.

**Cải thiện:**
```python
def train_with_validation(self, x_train, y_train, x_val, y_val,
                         epochs=10000, patience=20, ...):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # training...
        
        # validation
        val_loss, val_acc = self.evaluate(x_val, y_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            self.save('best_model.pkl')  # checkpoint
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    best_model = Model.load('best_model.pkl')
    return best_model
```

**Thuật ngữ:**
- **Overfitting:** Train loss ↓ nhưng val loss ↑ → generalization yếu.
- **Early stopping:** Dừng khi val metric không improve n epochs.
- **Checkpoint:** Lưu best model during training.
- **Generalization gap:** val_loss - train_loss → đo overfitting.

---

### 9. Learning Rate Schedulers (Ưu tiên 3.3)

**Ví dụ: Exponential Decay**
```python
class LRScheduler_Exponential:
    def __init__(self, initial_lr: float, decay_rate: float = 0.95):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.decay_rate ** epoch)
```

**Trong train loop:**
```python
scheduler = LRScheduler_Exponential(0.01, 0.95)
for epoch in range(epochs):
    self.optimizer.learning_rate = scheduler.get_lr(epoch)
    # ... training ...
```

**Thuật ngữ:**
- **Warm-up:** Tăng lr từ 0 → lr₀ dalam epoch đầu.
- **Cosine annealing:** lr(t) = lr₀ * (1 + cos(πt/T)) / 2 → smooth decay.

---

## PHẦN 4: LỖI LOW (TIER 4) — NICE-TO-HAVE

### 10. Batch Normalization (Ưu tiên 4.1)

```python
class BatchNormalization:
    """Normalize activations qua batch → faster training, higher LR."""
    def __init__(self, n_inputs: int, momentum: float = 0.9):
        self.momentum = momentum
        self.gamma = np.ones((1, n_inputs))  # scale
        self.beta = np.zeros((1, n_inputs))   # shift
        self.moving_mean = np.zeros((1, n_inputs))
        self.moving_var = np.ones((1, n_inputs))
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Update moving averages
            self.moving_mean = (self.momentum * self.moving_mean +
                               (1 - self.momentum) * batch_mean)
            self.moving_var = (self.momentum * self.moving_var +
                              (1 - self.momentum) * batch_var)
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + 1e-8)
        else:
            # Inference: dùng moving averages
            x_norm = (x - self.moving_mean) / np.sqrt(self.moving_var + 1e-8)
        
        return self.gamma * x_norm + self.beta
```

**Thuật ngữ:**
- **Internal covariate shift:** Phân phối input thay đổi qua training → chậm.
- **Batch normalization:** Chuẩn hóa mỗi layer activation → ổn định, nhanh.
- **Exponential moving average:** Trung bình weighted recent values → estimate population stats.

---

### 11. Dropout (Ưu tiên 4.2)

```python
class Dropout:
    """Randomly deactivate neurons → regularization."""
    def __init__(self, drop_rate: float = 0.2):
        self.drop_rate = drop_rate
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mask = np.random.binomial(1, 1 - self.drop_rate, x.shape)
            return x * mask / (1 - self.drop_rate)  # scale
        else:
            return x  # no dropout in inference
```

**Thuật ngữ:**
- **Dropout:** Randomly set activations → 0 (during training).
- **Co-adaptation:** Units học depend on nhau → overfit; Dropout breaks it.

---

## BẢNG TÓMO THUẬT NGỮ CHUYÊN NGÀNH

| Thuật ngữ | Định nghĩa | Liên quan |
|-----------|-----------|----------|
| **Feedforward** | Data → layer 1 → ... → output (no cycle) | Architecture |
| **Backpropagation** | Tính ∂L/∂w bằng chain rule ngược | Training |
| **Gradient flow** | Sự lan truyền ∂L qua layers | Backprop |
| **Overfitting** | Train loss ↓, test loss ↑ | Regularization, early stopping |
| **Underfitting** | Train & test loss cao | Increase model capacity |
| **Batch size** | Số samples per update | Trade-off: noise vs efficiency |
| **Epoch** | 1 lần đi qua toàn bộ dataset | Training loop |
| **Learning rate (α)** | Bước size trong gradient descent | Optimization |
| **Momentum** | Accumulate past gradients | Speed up convergence |
| **Adaptive LR** | Learning rate khác nhau per-parameter | AdaGrad, RMSProp, Adam |
| **Regularization** | Penalize large weights | L1, L2, Dropout |
| **Numerical stability** | Tránh NaN/Inf trong computation | Clipping, normalization |
| **Serialization** | Lưu object thành bytes/file | Save/Load |
| **Checkpoint** | Lưu best model during training | Early stopping |
| **Gradient clipping** | ||∇|| → max_norm | Prevent exploding gradient |
| **Xavier/He init** | Weight initialization cho layer | Prevent vanishing/exploding grad |
| **Batch normalization** | Normalize layer input qua batch | Faster training |
| **Dropout** | Randomly deactivate neurons | Regularization |
| **Early stopping** | Dừng khi val metric không improve | Prevent overfitting |
| **Learning rate schedule** | Giảm LR theo epoch/iteration | Better convergence |

---

## PRIORITIZED ACTION PLAN

### TIER 1 (Critical — NGAY):
- ✅ Sửa Softmax+CCE (done)
- ✅ Accuracy state (done)
- **→ Implement Model.save()/load()/create()** ← TOP PRIORITY
- Ensure backward signature consistency

### TIER 2 (High — trong 2 tuần):
- Add Xavier/He weight initialization + seed control
- Standardize Optimizer state (current_lr everywhere)
- Implement mini-batch training with shuffling
- Add batch_size + validation split parameters

### TIER 3 (Medium — trong 1 tháng):
- Gradient clipping
- Early stopping + checkpoint best model
- Learning rate schedulers (exponential, step, cosine)
- Unit tests (gradient check, shape validation, toy problem convergence)

### TIER 4 (Nice-to-have — future):
- Batch normalization layer
- Dropout layer
- Callbacks system
- Logging/TensorBoard integration

---

## Kết luận

Mã hiện tại là **foundation tốt** cho learning ANN, nhưng **PHẢI** thêm:
1. **Save/Load/Create** (persistence) → sử dụng trained models
2. **Numerical fixes** (Softmax+CCE) → correct gradients
3. **Training robustness** (early stopping, validation) → avoid overfitting
4. **Mini-batch support** (shuffling, batch_size) → real-world training

Sau đó có thể explore advanced topics (BatchNorm, Dropout, attention mechanisms).

