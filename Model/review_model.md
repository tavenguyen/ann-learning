# Review & Hướng dẫn Thiết kế ANN Model Chuyên nghiệp

Tài liệu này đánh giá chi tiết cấu trúc mã nguồn trong `Model/model.py`, tập trung vào cơ chế Registry, Serialization (Config) và các lỗi logic cần khắc phục để đạt chuẩn thiết kế chuyên nghiệp.

---

## 1. Phân tích cơ chế AppRegistry & Decorator

### Tư duy thiết kế (Design Philosophy)
Việc sử dụng `AppRegistry` kết hợp với `Decorator` là một kỹ thuật **Meta-programming** mạnh mẽ. Nó giúp giải quyết bài toán: *"Làm sao để hệ thống biết được có những lớp nào tồn tại mà không cần khai báo thủ công trong một danh sách dài dặc?"*

*   **Registry Pattern:** Tạo ra một "sổ hộ khẩu" trung tâm lưu trữ ánh xạ giữa tên định danh (chuỗi) và đối tượng lớp (Class object).
*   **Decorator:** Cho phép đăng ký lớp ngay tại thời điểm định nghĩa, giúp mã nguồn có tính **Decoupling** (giảm sự phụ thuộc) tuyệt vời. Bạn có thể thêm một Activation mới ở một file khác hoàn toàn, chỉ cần dùng `@CLASS_REGISTRY.register`, nó sẽ tự động có mặt trong hệ thống.

### Giải thích cú pháp & logic

#### Hàm `register(self, cls)`
```python
def register(self, cls):
    name = cls.__name__ # Lấy tên lớp gốc (vd: "DenseLayer")
    if name in self._registry:
        raise ValueError(f"Đã tồn tại class '{name}'")
    self._registry[name] = cls # Lưu Class Object vào dict
    return cls # Trả lại lớp gốc để Python tiếp tục xử lý
```
*   **Tại sao phải `return cls`?** Nếu không trả về lớp, biến định nghĩa lớp đó sẽ trở thành `None`. Decorator thực chất là một hàm bọc: `DenseLayer = register(DenseLayer)`.
*   **Tính an toàn:** Việc kiểm tra `if name in self._registry` giúp tránh tình trạng ghi đè lớp âm thầm, một lỗi cực kỳ khó debug trong các hệ thống lớn.

#### Hàm `get(self, name, *args, **kwargs)`
```python
def get(self, name, *args, **kwargs):
    cls = self._registry.get(name)
    if not cls:
        raise KeyError(f"Không tìm thấy class '{name}'")
    return cls(*args, **kwargs) # Khởi tạo instance ngay lập tức
```
*   **`*args, **kwargs`:** Đây là "vũ khí" của Master Python. Nó cho phép truyền mọi tham số từ cấu hình vào hàm `__init__` của lớp mà không cần biết trước lớp đó yêu cầu gì.

---

## 2. Kiểm tra sai sót trong hàm `getConfig`

Hiện tại, cơ chế lưu cấu hình (Serialization) đang gặp các lỗi "chết người" sau:

### Lỗi 1: Mâu thuẫn tên định danh (Naming Mismatch)
Registry lưu lớp bằng `cls.__name__` (ví dụ: `Activation_ReLU`). Tuy nhiên, hàm `getConfig` lại trả về một chuỗi khác (ví dụ: `ActivationReLU`).
*   **Hậu quả:** Khi gọi `createFromConfig`, Registry sẽ tìm `ActivationReLU` và báo lỗi `KeyError` vì nó chỉ biết lớp tên là `Activation_ReLU`.
*   **Sửa lỗi:** Chuỗi `className` trong `getConfig` phải trùng khớp tuyệt đối với tên lớp thực tế.

### Lỗi 2: Thiếu sót tham số và hàm (Omissions)
1.  **Thiếu hoàn toàn `getConfig`:** Các lớp `Activation_Softmax`, `Regularization_L1`, `Regularization_L2`, `Layer_Dropout` chưa có hàm này. Mô hình sẽ crash khi gọi `save`.
2.  **Thiếu cấu hình lồng (Nested Config):** `DenseLayer.getConfig` hiện chỉ lưu số n_neurons. Nó **bỏ quên** cấu hình của `weight_regularizer` và `bias_regularizer`.
3.  **Lỗi logic trong Accuracy:** `Accuracy_CategoricalClassification.getConfig` đang gọi `self.tolerance`, nhưng lớp này không hề có thuộc tính đó (chỉ lớp Regression mới có).

### Lỗi 3: Sai lệch tên tham số (Parameter Mismatch)
Hàm `__init__` của Optimizer dùng `learning_rate`, nhưng `getConfig` lại trả về `learningRate`. Khi nạp lại, Python sẽ báo lỗi: *`__init__() got an unexpected keyword argument 'learningRate'`*.

---

## 3. Hướng dẫn xây dựng `createobjectfromconfig` chuyên nghiệp

Hàm hiện tại trong code chỉ là bản nháp và sẽ không chạy được với các tham số. Một hàm "chuẩn Master" cần xử lý được sự **đệ quy** (recursive instantiation).

### Tư duy xây dựng:
1.  Lấy `className` để biết lớp cần tạo.
2.  Chuẩn bị bộ tham số (`kwargs`).
3.  **Quan trọng:** Nếu trong tham số có một dictionary chứa `className`, đó là một đối tượng lồng (ví dụ: Regularizer lồng trong Layer). Phải gọi đệ quy chính hàm này để tạo đối tượng con trước.
4.  Gọi `CLASS_REGISTRY.get(name, **kwargs)`.

### Code mẫu gợi ý:
```python
@staticmethod
def createobjectfromconfig(config):
    if config is None: return None
    
    class_name = config.get("className")
    # Lấy các tham số còn lại, loại bỏ className
    kwargs = {k: v for k, v in config.items() if k != "className"}
    
    # Xử lý đệ quy cho các đối tượng lồng (ví dụ: regularizers)
    for k, v in kwargs.items():
        if isinstance(v, dict) and "className" in v:
            kwargs[k] = Model.createobjectfromconfig(v)
            
    return CLASS_REGISTRY.get(class_name, **kwargs)
```

---

## 4. Các lỗi Logic khác cần sửa ngay trong code

Mặc dù bạn không yêu cầu sửa code, nhưng với tư cách Master, tôi phải chỉ ra các lỗi khiến mô hình không thể chạy đúng:

1.  **Thụt lề sai (Critical):** Trong hàm `get_regularization_loss`, lệnh `return reg_loss` bị thụt lề vào trong vòng lặp `for`. Kết quả là nó chỉ tính loss cho layer đầu tiên rồi thoát luôn.
2.  **Lỗi chính tả trong DataLoader:** 
    *   Lưu là `self.suffle` nhưng dùng là `self.shuffle`.
    *   Trong `__iter__`, bạn tạo biến cục bộ `X_shuffled` nhưng đoạn dưới lại gọi `self.X_shuffled`. Điều này gây lỗi `AttributeError`.
3.  **Lỗi logic trong Model:** Hàm `setParameters` đang dùng `self.trainableLayers` (viết hoa chữ L), nhưng hàm `finalize` lại tạo ra `self.trainable_layers` (có dấu gạch dưới).

---

### Tổng kết kiến thức
Để viết built-in model chuyên nghiệp, bạn cần nắm vững:
1.  **Registry Pattern:** Quản lý đối tượng động.
2.  **Serialization:** Khả năng biến đối tượng thành dữ liệu (dict/json) và ngược lại một cách toàn vẹn (đủ tham số, đúng tên).
3.  **Factory Pattern:** Sử dụng các hàm `create...` để khởi tạo đối tượng từ dữ liệu thô.

Hãy cập nhật các hàm `getConfig` và sửa lỗi đặt tên tham số để mô hình của bạn có thể Lưu/Tải một cách hoàn hảo!
