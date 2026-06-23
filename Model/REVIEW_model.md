# Code Review — Model/model.py

Tóm tắt ngắn: file chứa nhiều optimizer và activation/basic layer nhưng có một số lỗi logic nghiêm trọng, bất nhất API và chỗ thiếu tính bền vững. Dưới đây là danh sách lỗi, cách sửa cụ thể và các ý tưởng cải thiện tư duy thiết kế.

---

## 1) Lỗi nghiêm trọng (bắt buộc sửa ngay)

1.1. Cập nhật tham số dùng nhầm thuộc tính `self.*` thay `layer.*` (sẽ không thay đổi layer thực tế).
- Vị trí: `Optimizer_SGD_Decay.update_params` (dùng `self.weights/self.biases`), `Optimizer_AdaGrad.update_params` (dùng `self.weights/self.biases`).
- Hậu quả: optimizer không cập nhật weights của layer.

1.2. Công thức cập nhật sai cho AdaGrad
- Vị trí: dòng dùng `self.weights -= self.learning_rate * (np.sqrt(layer.weight_cache) + self.epsilon) * layer.dweights`
- Sai: nhân với sqrt(cache) thay vì chia và cập nhật thuộc tính sai đối tượng.
- Sửa đúng:

```python
layer.weight_cache += layer.dweights ** 2
layer.bias_cache += layer.dbiases ** 2
layer.weights -= self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
layer.biases  -= self.learning_rate * layer.dbiases  / (np.sqrt(layer.bias_cache)  + self.epsilon)
```

1.3. Lỗi chính tả và tên biến
- `intial_lr` -> `initial_lr` (Optimizer_SGD_Decay). Gây khó hiểu.

1.4. pre_update_params / post_update_params làm không cần thiết hoặc vô nghĩa
- Nhiều chỗ gán `self.learning_rate = self.learning_rate` — không có tác dụng. Nếu mục tiêu là tính learning rate theo decay, chỉ tính và lưu `current_lr`.

---

## 2) Thiết kế API bất nhất (khuyến nghị sửa)

2.1. Activation.backward nên luôn trả về `dinputs`
- `Activation_Linear.backward` trả về, `Activation_ReLU.backward` không trả về — gây bất nhất cho pipeline.
- Hãy chuẩn hóa: luôn `return self.dinputs`.

2.2. `DenseLayer` thiếu API: forward, backward, lưu outputs, dweights, dbiases
- Hiện chỉ chứa weights/biases. Để dùng trong training loop cần:
  - `forward(inputs)` -> lưu inputs và outputs
  - `backward(dvalues)` -> tính dweights, dbiases, dinputs

Ví dụ tối thiểu:

```python
class DenseLayer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
```

2.3. Velocities / caches phải là thuộc tính per-layer, khởi tạo bằng `np.zeros_like(layer.weights)`
- Một số optimizer dùng `self.weight_velocity = 0` (global scalar) — sai.

2.4. Momentum implementation consistency
- Hai kiểu phổ biến:
  1) v = beta * v + learning_rate * dW; W -= v
  2) v = beta * v + (1 - beta) * dW; W -= learning_rate * v
- Chọn 1 và dùng nhất quán. Khuyến nghị (1) vì đơn giản: maintain velocities in same units as weight updates so W -= v.

---

## 3) Những chỗ cần sửa cụ thể (dạng checklist)

- [ ] Fix Optimizer_SGD_Decay: use `layer.weights/layer.biases`, fix learning rate formula.
- [ ] Fix Optimizer_AdaGrad: use `layer.*`, divide by sqrt(cache)+eps (not multiply).
- [ ] Fix Optimizer_SGD_Momentum: apply velocity consistently and avoid double-multiplying learning_rate.
- [ ] Replace scalar `weight_velocity/bias_velocity` defaults with `np.zeros_like` per-layer.
- [ ] Make Activation_ReLU.backward return dinputs.
- [ ] Implement forward/backward in DenseLayer (see snippet above).
- [ ] Rename `intial_lr` -> `initial_lr`.

---

## 4) Code hygiene, style và best practices

- Thêm type hints và docstrings cho lớp và phương thức quan trọng.
- Thêm seed (hoặc cho phép truyền seed) cho reproducibility trong thử nghiệm.
- Khởi tạo weights với small scale (e.g., *0.01 hoặc theo He/Xavier tùy activation) thay vì np.random.randn thô.
- Đặt epsilon mặc định (1e-7) là hợp lý; document lý do.
- Kiểm tra shape khi tính dweights/dbiases để tránh broadcasting lỗi.
- Tách optimizer base class để chia sẻ iterations, hooks (pre/post), và contract `update_params(layer)` — giảm duplication.

---

## 5) Testing và Verification

- Viết unit tests nhỏ cho mỗi optimizer để kiểm tra: 1) sau 1 bước, weights thay đổi theo hướng mong đợi; 2) decay/momentum/caches tích luỹ đúng.
- Test forward/backward của DenseLayer bằng gradient check (finite differences) trên loss scalar nhỏ để verify dweights.

---

## 6) Ý tưởng nâng cao / phát triển tư duy

- Xây dựng interface `Layer` và `Optimizer` (abstract base classes) để enforce contract.
- Hỗ trợ mini-batch training, callbacks (logging, lr scheduling), checkpoint save/load weights.
- Thêm các initializer (He/Xavier/Uniform) và cho phép chọn initializer khi tạo layer.
- Hỗ trợ vectorized training loop: model.train(X, y, epochs, batch_size, ...) sử dụng layers với forward/backward.
- Thực hành TDD: viết tests trước khi sửa optimizer để bắt được regressions.

---

## 7) Gợi ý sửa nhanh (small PR)

1. Thêm forward/backward cho DenseLayer.
2. Sửa AdaGrad + RMSProp + Momentum để dùng layer.* và caches/velocities per-layer.
3. Chuẩn hóa Activation.backward trả về self.dinputs.
4. Thêm một base Optimizer để lưu iterations + pre/post hooks.

---

## 8) Nếu muốn, tôi có thể

- Tạo PR sửa tự động các lỗi ở mục 1 và 2 (bao gồm unit tests cơ bản). (Hãy trả lời "hãy sửa" để tôi bắt đầu.)
- Hoặc chỉ tạo file review này — nếu ok, hãy cho biết muốn sửa tự động hay từng bước.

---

Cuối: giữ phong cách code đơn giản, mỗi optimizer chỉ làm một việc rõ ràng, API nhất quán là chìa khoá để mở rộng an toàn.
