# Code review — Model/model.py

## Tóm tắt ngắn
File model.py chứa một ANN cơ bản hoạt động, nhưng có nhiều lỗi logic, sai shape, và thiếu chuẩn hóa API giữa các thành phần (loss/activation/optimizer). Dưới đây liệt kê các vấn đề chính và đề xuất sửa theo thứ tự ưu tiên.

---

## Lỗi nghiêm trọng (cần sửa gấp)

1. Softmax + Categorical Cross-Entropy
   - Clip đang áp lên logits (y_pred) thay vì probabilities.
   - Indexing sai: `y_pred_clipped[samples, y_true]` phải là `y_pred_clipped[range(samples), y_true]`.
   - Khi y_true là one-hot, xử lý shape không đúng (keepdims không cần thiết) và tính loss sai.
   - backward signature không tương ứng với cách Model.backward gọi (Model gọi `loss.backward(y_pred, y_true)` nhưng loss.backward định nghĩa `backward(self, dvalues, y_true)`).
   - Hậu quả: gradient truyền ngược bị sai, đào tạo không hội tụ.

2. Model.backward / Loss API inconsistent
   - Không có interface thống nhất cho Loss: forward phải trả losses per-sample, backward phải nhận probabilities (hoặc logits) và y_true, sau đó set `self.dinputs`.
   - Hiện tại các lớp loss khác nhau có signature khác nhau -> dễ gây lỗi.

3. Accuracy và trạng thái tích lũy
   - `Accuracy.calculate` dùng `accumulated_sum` và `accumulated_count` nhưng không được khởi tạo trong __init__ nên sẽ lỗi nếu không gọi `new_pass()` trước.
   - `Accuracy_CategoricalClassification.calculate` trả về mảng boolean thay vì một scalar (mean) và không cập nhật state tích lũy.

---

## Vấn đề về optimizer và numeric

4. pre_update_params vô nghĩa ở nhiều optimizer (gán lại chính biến), thiếu chuẩn hóa `current_lr`/`initial_lr` giữa các optimizer.
5. Momentum/RMSProp/Adam: tên biến và bias-correction, decay handling không nhất quán (ví dụ `current_lr` vs `learning_rate`).
6. Adam: tên cache và cách bias-correction có thể gây lỗi nếu iterations không được cập nhật đúng trước khi dùng.
7. Không có gradient clipping; epsilon nhỏ/nhất quán nên dùng 1e-8.

---

## Tính nhất quán activation / layer API

8. Activation_ReLU.forward không gán `self.output` (một số activation gán, một số không) -> thiếu nhất quán.
9. DenseLayer sử dụng np.random.randn cho init weights; nên có tùy chọn Xavier/He và tham số seed.
10. Model.forward dùng try/except TypeError để gọi forward với/không training param — nên chuẩn hóa tất cả layer có signature `forward(self, inputs, training=False)`.

---

## Thiết kế & tính năng đề xuất

- Thống nhất API cho Layer/Activation/Loss/Optimizer (docstring nêu shapes).  
- Hỗ trợ mini-batch, data shuffling, `batch_size`, và seed reproducibility.  
- Thêm unit tests: shape tests, numerical gradient check, toy-problem convergence (AND/XOR, Iris).  
- Thêm weight init (Xavier/He), regularization (L1/L2), dropout, batchnorm, learning-rate schedulers, early stopping.

---

## Hành động ưu tiên đề xuất

1. Sửa Softmax+CCE forward/backward và chỉnh Model.backward để lấy dvalues đúng.
2. Chuẩn hóa Accuracy API và khởi tạo accumulators trong __init__.
3. Chuẩn hoá pre_update/update/post_update trong optimizer (dùng current_lr consistently) và sửa Adam bias-correction nếu cần.
4. Thêm unit tests nhỏ để đảm bảo grad và shapes đúng.

---

Nếu muốn, sẽ tạo patch sửa 1) Softmax+CCE + Model.backward và 2) Accuracy init — chọn một trong các mục sau để bắt đầu:
- A: Sửa Softmax+CCE + Model.backward (ưu tiên)
- B: Chuẩn hóa Accuracy và khởi tạo state
- C: Weight init (He/Xavier) + seed
- D: Tất cả các sửa trên trong 1 PR

