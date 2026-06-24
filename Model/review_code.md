# Code Review Chuyên sâu — Model/model.py

## Tóm tắt Executive Summary
File model.py triển khai một **Feedforward Neural Network (FNN)** với backpropagation, nhưng gặp nhiều vấn đề về **numerical stability**, **API inconsistency**, **state management**, và thiếu **persistence layer (serialization)**. Các lỗi này ảnh hưởng trực tiếp đến **gradient flow** và **model reproducibility**.

---

## 1. Lỗi Numerical & Gradient Flow (Critical — Priority 1)

### 1.1 Softmax + Categorical Cross-Entropy Bug
**Vấn đề:**
- **Clip sai vị trí:** Áp dụng np.clip trên logits thay vì post-softmax probabilities. Điều này **không ngăn underflow** trong log-sum-exp.
- **Indexing lỗi:** `y_pred_clipped[samples, y_true]` – sai shape indexing; cần `y_pred_clipped[range(samples), y_true]`.
- **One-hot encoding inconsistency:** Khi y_true là one-hot matrix, phép tính `np.sum(probs_clipped * y_true, axis=1)` trả về shape (N,) nhưng không xoá chiều nên keepdims có thể gây mismatch.
- **API signature mismatch:** backward nhận y_pred nhưng cần dvalues từ loss output (probs); tạo confuse trong gradient backpropagation.

**Hậu quả:**
- **Numerical instability:** NaN/Inf trong training do log(0) hoặc underflow.
- **Incorrect gradient:** dinputs được tính từ logits thay vì probs → sai **backpropagation**.
- **Non-convergence:** Loss không giảm hoặc oscillate không kiểm soát.

**Thuật ngữ chuyên ngành:** 
- **Softmax:** Hàm activation chuẩn hóa output thành probability distribution (sum = 1).
- **Categorical Cross-Entropy (CCE):** Loss function cho multi-class classification: L = -Σ y_true * log(y_pred).
- **Numerical stability:** Kỹ thuật tránh overflow/underflow bằng log-sum-exp trick.
- **Gradient flow:** Sự lan truyền đạo hàm từ output → input qua mạng (backprop).

### 1.2 Model.backward & Loss API Inconsistency
**Vấn đề:**
- Loss không có base class hoặc protocol để enforce uniform interface.
- MSE.backward(y_pred, y_true) vs Softmax+CCE.backward(dvalues, y_true) → signature khác nhau.
- Model.backward gọi `loss.backward(y_pred, y_true)` nhưng một số loss subclass kỳ vọng dvalues (post-activation gradients).

**Thuật ngữ:**
- **Loss interface:** Hợp đồng (contract) giữa Model và Loss class.
- **Backpropagation:** Thuật toán tính gradient bằng cách đi ngược qua graph.

---

## 2. State Management & Accumulator Bugs (Critical — Priority 1)

### 2.1 Accuracy Stateful Accumulation
**Vấn đề:**
- Accuracy.__init__ không khởi tạo accumulated_sum/accumulated_count → **AttributeError** nếu gọi calculate() trước new_pass().
- Accuracy_CategoricalClassification không kế thừa Accuracy → không dùng được accumulated mechanics.
- calculate() trả về boolean array thay vì scalar → không thể mean() trực tiếp.

**Hậu quả:**
- **Epoch-wise accuracy** không được tích lũy đúng → metrics sai lệch.
- **Validation loop** có thể fail hoặc báo cáo metrics không chính xác.

**Thuật ngữ:**
- **Accumulation:** Tích lũy các giá trị qua nhiều mini-batches để tính metric epoch-level.
- **State initialization:** Khởi tạo state ban đầu để tránh UB (Undefined Behavior).

### 2.2 Optimizer State Inconsistency
**Vấn đề:**
- Biến learning rate gọi khác nhau: Optimizer_SGD dùng `learning_rate`, Optimizer_SGD_Decay dùng `current_lr`.
- pre_update_params() nhiều chỗ chỉ `self.x = self.x` → vô nghĩa, thiếu decay calculation.
- Adam bias-correction dùng (self.iterations + 1) nhưng iterations được increment ở post_update → timing sai.

**Thuật ngữ:**
- **Learning rate decay:** Giảm learning rate theo epoch/iteration để converge tốt hơn.
- **Bias correction:** Hiệu chỉnh exponential moving average khi iterations nhỏ (trong Adam).
- **First-order moment & second-order moment:** Ước lượng trung bình gradient (m) và bình phương gradient (v) trong Adam.

---

## 3. Layer Initialization & Weight Distribution (High — Priority 2)

### 3.1 Weight Initialization
**Vấn đề:**
- `np.random.randn(n_inputs, n_neurons)` sử dụng chuẩn N(0,1) → **không mang tính chất activation function**.
- Không có tùy chọn Xavier (Glorot) hoặc He initialization → **vanishing/exploding gradient** khi network sâu.
- Không có seed control → **non-reproducible** training runs.

**Thuật ngữ:**
- **Xavier/Glorot init:** Khởi tạo weights ∼ U[-√(6/(n_in+n_out)), √(6/(n_in+n_out))] để maintain variance qua layers.
- **He init:** Khởi tạo weights ∼ N(0, √(2/n_in)) cho ReLU (vì ReLU kills ~50% gradient).
- **Vanishing gradient:** Gradient → 0 khi backprop qua layers sâu (activation như sigmoid/tanh).
- **Exploding gradient:** Gradient → ∞ (NaN) khi weights quá lớn.

---

## 4. Regularization & Gradient Penalty (Medium — Priority 2)

### 4.1 L1/L2 Regularization Implementation
**Vấn đề hiện tại (đã được cải thiện):**
- DenseLayer.backward đã thêm gradient regularization.
- Nhưng Model.train/evaluate chưa report regularization loss chi tiết.

**Thuật ngữ:**
- **L1 regularization (Lasso):** Loss += λ₁ * Σ|w| → tạo sparsity (weights → 0).
- **L2 regularization (Ridge):** Loss += λ₂ * Σw² → shrink weights nhưng không sparse.
- **Regularization loss:** Penalty term để tránh overfitting bằng cách penalize large weights.
- **Weight decay:** Trong SGD, bằng cách trừ learning_rate * λ * w trong weight update.

---

## 5. Model Persistence — Thiếu Save/Load/Create (Critical — Priority 1)

### 5.1 Không có Model Serialization/Deserialization
**Vấn đề:**
- **Không có save():** Mô hình đã train không thể lưu → mất thời gian train nếu thoát chương trình.
- **Không có load():** Không thể nạp mô hình đã save → không reproducible.
- **Không có model.create():** Không có method để tạo model từ config (architecture definition).
- **Stateless weights:** Sau train, nếu process dies, tất cả weights đều mất.

**Hậu quả:**
- **No checkpointing:** Không lưu best model during training.
- **Loss of trained parameters:** Mỗi lần chạy code phải retrain từ đầu.
- **Deployment impossible:** Không thể load mô hình cho production inference.

**Thuật ngữ:**
- **Serialization/Deserialization:** Lưu trữ object (weights, architecture) thành file và ngược lại.
- **Checkpoint:** Lưu state của model tại các điểm nhất định (ví dụ best validation loss).
- **Model architecture:** Danh sách layers, sizes, activations.
- **Model weights/parameters:** Giá trị thực tế của w, b sau training.

### 5.2 Recommended Save/Load Interface
```python
# Save model
model.save(filepath='model.pkl')  # or .npz, .h5, etc.

# Load model
loaded_model = Model.load(filepath='model.pkl')

# Inference
predictions = loaded_model.predict(X_test)
```

---

## 6. Training Loop & Validation Protocol (Medium — Priority 2)

### 6.1 Mini-batch & Data Shuffling
**Vấn đề:**
- Model.train() dùng full-batch gradient descent (toàn bộ dataset một lần).
- Không có shuffling → ordering bias, convergence chậm.
- Không có batch_size parameter → không flexible cho large datasets.

**Thuật ngữ:**
- **Batch size:** Số samples dùng trong một forward-backward pass.
- **Epoch:** Một lần đi qua toàn bộ training set.
- **Shuffling:** Random reorder samples mỗi epoch để tránh ordering bias.
- **Mini-batch GD:** Update weights sau mỗi batch (trade-off giữa noise & efficiency).

### 6.2 Validation & Early Stopping
**Vấn đề:**
- Model.evaluate() chỉ report loss & accuracy, không có early stopping.
- Không tracking best model during training.
- Không splitting train/val/test sets.

**Thuật ngữ:**
- **Overfitting:** Model học noise → performance tốt trên train nhưng tệ trên test.
- **Early stopping:** Dừng training khi validation loss không giảm n epochs liên tiếp.
- **Generalization gap:** Difference giữa train loss và validation loss.

---

## 7. Activation Functions & Layer API (Medium — Priority 2)

### 7.1 Activation Consistency
**Vấn đề:**
- Activation_ReLU.forward() không gán self.output → inconsistent với Activation_Linear.
- Model.forward() dùng try/except TypeError để gọi forward(inputs, training) → fragile.

**Thuật ngữ:**
- **Activation function:** Non-linear transformation f(x) = max(0, x) (ReLU), etc.
- **Forward pass:** Compute output từ input qua layers (inference).
- **Backward pass:** Compute gradient w.r.t. weights từ loss (training).

---

## 8. Numerical Stability & Gradient Clipping (Medium — Priority 3)

### 8.1 Gradient Explosion
**Vấn đề:**
- Không có gradient clipping → **exploding gradient** khi ||∇|| quá lớn.
- Epsilon trong Adam/RMSProp là 1e-7 → có thể quá nhỏ trên devices khác nhau.

**Thuật ngữ:**
- **Gradient clipping:** Giới hạn ||∇|| ≤ max_norm để tránh exploding.
- **Epsilon (ε):** Số rất nhỏ để avoid division by zero.

---

## 9. Testing & Documentation (Low — Priority 3)

### 9.1 Unit Tests
**Vấn đề:**
- Không có tests: shape validation, gradient checking, convergence on toy problems.
- Không có docstrings chi tiết → khó dùng API.

**Thuật ngữ:**
- **Gradient checking:** Số mở numerically (finite differences) để verify analytic gradient.
- **Shape inference:** Validate input/output shapes qua layers.
- **Toy problems:** Simple datasets (AND, XOR, Iris) để test convergence.

---

## 10. Optimization Algorithms — Inconsistencies (Medium — Priority 2)

### 10.1 Optimizer State Management
**Vấn đề:**
- SGD_Momentum tính velocity sai (vi += lr * dw thay vì vi = beta*vi - lr*dw standard).
- AdaGrad cache không reset → learning rate luôn giảm (không phù hợp lâu dài).
- RMSProp parameter p (retention ratio) có thể confuse với Momentum's beta.

**Thuật ngữ:**
- **Momentum:** Accumulate past gradients để tăng tốc hội tụ, avoid local minima.
- **Adaptive learning rate:** Learning rate khác nhau per-parameter (AdaGrad, RMSProp, Adam).
- **Exponential moving average:** Weighted average với recent gradients được weight cao hơn.
- **Bias-correction term:** Trong Adam, điều chỉnh vì biased estimator khi iterations nhỏ.

---

## Hành động ưu tiên (Prioritized Action Items)

### Tiers:

**TIER 1 (Critical — Phải sửa ngay):**
1. ✅ Sửa Softmax+CCE forward/backward signature
2. ✅ Chuẩn hóa Loss API (backward signature)
3. ✅ Fix Accuracy initialization & inheritance
4. ⏳ Implement Model.save() / Model.load() (pickle/numpy serialization)

**TIER 2 (High — Nên sửa sớm):**
5. ⏳ Weight initialization (Xavier/He + seed)
6. ⏳ Chuẩn hóa Optimizer state variables (use current_lr everywhere)
7. ⏳ Fix Adam bias-correction timing
8. ⏳ Implement batch_size + data shuffling trong train()

**TIER 3 (Medium — Tối ưu hóa):**
9. ⏳ Gradient clipping implementation
10. ⏳ Early stopping + checkpoint best model
11. ⏳ Learning rate schedulers (exponential decay, step decay)
12. ⏳ Implement unit tests (gradient check, shape validation)

**TIER 4 (Nice-to-have — Tương lai):**
13. ⏳ Dropout layer, BatchNormalization
14. ⏳ Callbacks system
15. ⏳ TensorBoard integration

---

## Ví dụ Code: Save/Load Model

```python
import pickle
import json

# Save model
def save(self, filepath: str):
    """Serialize model architecture & weights."""
    model_state = {
        'layers': [
            {
                'type': layer.__class__.__name__,
                'weights': layer.weights if hasattr(layer, 'weights') else None,
                'biases': layer.biases if hasattr(layer, 'biases') else None,
                'config': {...}  # activation, regularizer config
            } for layer in self.layers
        ],
        'loss': self.loss.__class__.__name__,
        'optimizer': {...}
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_state, f)

# Load model
@staticmethod
def load(filepath: str):
    """Deserialize model."""
    with open(filepath, 'rb') as f:
        model_state = pickle.load(f)
    # Reconstruct layers, loss, optimizer từ state
    ...
```

---

## Từ ngữ chuyên ngành tóm tắt

| Khái niệm | Nghĩa | Liên quan |
|-----------|-------|----------|
| **Forward pass** | Tính output từ input qua layers | Inference, training |
| **Backward pass** | Tính gradient từ loss qua layers (backprop) | Training, optimization |
| **Gradient flow** | Sự lan truyền đạo hàm qua mạng | Backpropagation |
| **Numerical stability** | Tránh NaN/Inf trong tính toán | Clipping, normalization |
| **Overfitting** | Model fit noise → generalize tệ | Regularization, early stopping |
| **Gradient checking** | Verify analytic gradient với finite-difference | Testing |
| **Regularization** | Penalize large weights → tránh overfitting | L1, L2 |
| **Serialization** | Lưu object thành file | Save/Load |
| **Checkpoint** | Lưu best model during training | Validation |
| **Batch size** | Số samples/update | Mini-batch GD |

