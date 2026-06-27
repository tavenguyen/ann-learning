# [ANN] Mini-batch Training: Bản chất, công thức và tư duy OOP khi huấn luyện mạng Neural Network

## 1. Mục tiêu

Tài liệu này giải thích:

* Mini-batch Training là gì.
* Sự khác nhau giữa Full-batch Gradient Descent, Stochastic Gradient Descent và Mini-batch Gradient Descent.
* Các khái niệm `epoch`, `batch`, `batch_size`, `iteration`, `steps_per_epoch`.
* Vì sao cần shuffle dữ liệu trước mỗi epoch.
* Cách tính Loss và Accuracy đúng khi train theo batch.
* Vì sao optimizer nên update sau mỗi batch.
* Tư duy xây dựng `DataLoader` hoặc `BatchIterator`.
* Tư duy tích hợp mini-batch vào class `Model`.
* Những lỗi thường gặp khi dùng mini-batch training trong ANN.

---

# 2. Vấn đề khi train toàn bộ dataset một lần

Khi mới học ANN, ta thường train như sau:

```python
for epoch in range(epochs):
    output = model.forward(X)

    loss = loss_function.calculate(
        output,
        y
    )

    model.backward(
        output,
        y
    )

    optimizer.update_params(...)
```

Ở đây, toàn bộ dataset `X` được đưa vào model một lần.

Nếu dataset nhỏ, cách này vẫn ổn.

Nhưng nếu dataset lớn, ví dụ:

```text
10,000 mẫu
100,000 mẫu
1,000,000 mẫu
```

thì việc forward và backward toàn bộ dataset một lúc sẽ gây ra các vấn đề:

```text
Tốn RAM.

Tính toán chậm.

Một lần update mất nhiều thời gian.

Không tận dụng tốt tính chất ngẫu nhiên của gradient.

Khó train trên dữ liệu lớn.
```

Vì vậy ta chia dataset thành nhiều phần nhỏ hơn.

Mỗi phần nhỏ gọi là một mini-batch.

---

# 3. Mini-batch là gì?

Mini-batch là một nhóm nhỏ các mẫu được lấy từ dataset để train trong một lần forward, backward và update.

Giả sử dataset có $N$ mẫu:

$$D=\{(x_1,y_1),(x_2,y_2),\ldots,(x_{N},y_{N})\}$$

Ta chia thành các batch nhỏ, mỗi batch có khoảng $B$ mẫu.

Trong đó:

* $N$ là tổng số mẫu.
* $B$ là `batch_size`.

Ví dụ:

```python
N = 1000
batch_size = 32
```

Số batch trong một epoch xấp xỉ:

$$\mathrm{stepsPerEpoch}=\lceil \frac{N}{B}\rceil$$

Với:

$$N=1000$$

$$B=32$$

ta có:

$$\mathrm{stepsPerEpoch}=\lceil \frac{1000}{32}\rceil=32$$

Nghĩa là một epoch có 32 lần update.

---

# 4. Epoch, Batch và Iteration

Ba khái niệm này rất quan trọng.

## Epoch

Một epoch nghĩa là model đã đi qua toàn bộ training dataset một lần.

Nếu dataset có 1000 mẫu, sau một epoch model đã nhìn thấy đủ 1000 mẫu.

## Batch

Batch là một phần nhỏ của dataset.

Ví dụ:

```text
Dataset có 1000 mẫu.

Batch size = 100.

Một epoch có 10 batch.
```

## Iteration

Một iteration thường tương ứng với một lần update parameters.

Trong mini-batch training:

```text
1 batch
→ forward
→ loss
→ backward
→ update
→ 1 iteration
```

Do đó:

$$\mathrm{iterations}=\mathrm{epochs}\times\mathrm{stepsPerEpoch}$$

Ví dụ:

```text
epochs = 10
steps_per_epoch = 32
```

thì:

$$\mathrm{iterations}=10\times32=320$$

Trong optimizer có decay, `iterations` nên tăng sau mỗi batch, không phải sau mỗi epoch.

---

# 5. Ba kiểu Gradient Descent

## 5.1 Full-batch Gradient Descent

Dùng toàn bộ dataset để tính gradient trong mỗi lần update.

Công thức Loss:

$$L=\frac{1}{N}\sum_{i=1}^{N}L_{i}$$

Gradient:

$$\nabla W=\frac{1}{N}\sum_{i=1}^{N}\nabla W_{i}$$

Tư duy:

```text
Một epoch chỉ update một lần.

Gradient ổn định.

Nhưng update chậm nếu dataset lớn.
```

Ưu điểm:

```text
Gradient ít nhiễu.

Loss giảm mượt hơn.
```

Nhược điểm:

```text
Tốn RAM.

Mỗi update rất chậm.

Không phù hợp với dataset lớn.
```

---

## 5.2 Stochastic Gradient Descent

Dùng một mẫu duy nhất để update.

Tức là:

$$B=1$$

Gradient:

$$\nabla W=\nabla W_{i}$$

Tư duy:

```text
Mỗi mẫu tạo một lần update.

Update rất thường xuyên.

Gradient rất nhiễu.
```

Ưu điểm:

```text
Tốn ít RAM.

Update nhanh theo từng mẫu.

Có tính ngẫu nhiên cao, đôi khi thoát khỏi vùng xấu tốt hơn.
```

Nhược điểm:

```text
Loss dao động mạnh.

Training không ổn định.

Không tận dụng tốt vectorization của NumPy hoặc GPU.
```

---

## 5.3 Mini-batch Gradient Descent

Dùng một nhóm nhỏ $B$ mẫu để update.

Gradient trên một batch:

$$\nabla W_{\mathrm{batch}}=\frac{1}{B}\sum_{i=1}^{B}\nabla W_{i}$$

Tư duy:

```text
Không dùng toàn bộ dataset.

Cũng không dùng từng mẫu riêng lẻ.

Dùng một nhóm mẫu nhỏ để cân bằng tốc độ và độ ổn định.
```

Ưu điểm:

```text
Tận dụng vectorization.

Tốn ít RAM hơn full-batch.

Gradient ổn định hơn SGD từng mẫu.

Update thường xuyên hơn full-batch.
```

Vì vậy mini-batch là cách train phổ biến nhất trong ANN.

---

# 6. So sánh nhanh

| Phương pháp  | Batch size | Số update trong một epoch | Gradient  | Tốc độ                | RAM      |
| ------------ | ---------: | ------------------------: | --------- | --------------------- | -------- |
| Full-batch   |        $N$ |                         1 | Ít nhiễu  | Chậm trên dataset lớn | Cao      |
| SGD từng mẫu |          1 |                       $N$ | Rất nhiễu | Nhiều update nhỏ      | Thấp     |
| Mini-batch   |        $B$ |        $\lceil N/B\rceil$ | Vừa phải  | Tốt                   | Vừa phải |

---

# 7. Vì sao cần shuffle dữ liệu?

Nếu dữ liệu có thứ tự, ví dụ:

```text
100 mẫu class 0
sau đó 100 mẫu class 1
sau đó 100 mẫu class 2
```

Nếu không shuffle, các batch đầu chỉ chứa class 0, các batch sau chỉ chứa class 1 hoặc class 2.

Khi đó gradient của mỗi batch bị lệch theo thứ tự dữ liệu.

Nên trước mỗi epoch, ta xáo trộn dataset:

```python
indices = np.random.permutation(N)

X_shuffled = X[indices]
y_shuffled = y[indices]
```

Tư duy:

```text
Mỗi batch nên đại diện tương đối cho toàn bộ dataset.

Shuffle giúp batch không bị phụ thuộc vào thứ tự dữ liệu ban đầu.
```

---

# 8. Có nên shuffle validation và test set không?

Training set nên shuffle.

Validation và test set không cần shuffle để đánh giá.

Với validation và test:

```text
Không update weights.

Chỉ forward và tính metric.

Thứ tự dữ liệu không ảnh hưởng đến Loss trung bình.
```

Có thể shuffle validation/test nếu muốn, nhưng không bắt buộc.

---

# 9. Last batch là gì?

Nếu $N$ không chia hết cho $B$, batch cuối sẽ nhỏ hơn.

Ví dụ:

```text
N = 1000
batch_size = 128
```

Ta có:

```text
7 batch đầu: 128 mẫu
batch cuối: 104 mẫu
```

Vì:

$$1000=7\times128+104$$

Batch cuối vẫn nên được train bình thường.

Nếu bỏ batch cuối, ta mất một phần dữ liệu mỗi epoch.

---

# 10. `drop_last` là gì?

`drop_last=True` nghĩa là bỏ batch cuối nếu batch đó không đủ `batch_size`.

Ví dụ:

```text
N = 1000
batch_size = 128
```

Nếu `drop_last=True`, chỉ dùng:

$$7\times128=896$$

mẫu.

104 mẫu cuối bị bỏ.

Trong ANN cơ bản, thường nên dùng:

```python
drop_last = False
```

để không mất dữ liệu.

`drop_last=True` hữu ích trong một số trường hợp cần batch size cố định, nhưng không bắt buộc trong bài học cơ bản.

---

# 11. Cách tạo mini-batch thủ công

```python
def create_batches(
    X,
    y,
    batch_size,
    shuffle=True,
    drop_last=False
):
    n_samples = len(X)

    if shuffle:
        indices = np.random.permutation(
            n_samples
        )
    else:
        indices = np.arange(
            n_samples
        )

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for start in range(
        0,
        n_samples,
        batch_size
    ):
        end = start + batch_size

        if end > n_samples and drop_last:
            break

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        yield X_batch, y_batch
```

Cách dùng:

```python
for X_batch, y_batch in create_batches(
    X,
    y,
    batch_size=32,
    shuffle=True
):
    output = model.forward(
        X_batch
    )

    loss = loss_function.calculate(
        output,
        y_batch
    )

    model.backward(
        output,
        y_batch
    )

    optimizer.update_params(...)
```

---

# 12. Tư duy `yield` trong Python

Hàm `create_batches()` dùng `yield`.

Điều đó có nghĩa mỗi lần lặp, hàm trả ra một batch.

Ví dụ:

```python
for X_batch, y_batch in create_batches(X, y, 32):
    ...
```

Mỗi vòng lặp nhận một phần dữ liệu.

Ưu điểm:

```text
Không cần tạo sẵn toàn bộ danh sách batch.

Tiết kiệm bộ nhớ.

Code gọn hơn.
```

---

# 13. DataLoader OOP

Có thể xây dựng một class `DataLoader` để quản lý mini-batch.

```python
import numpy as np

class DataLoader:
    def __init__(
        self,
        X,
        y,
        batch_size=32,
        shuffle=True,
        drop_last=False
    ):
        if len(X) != len(y):
            raise ValueError(
                "X và y phải có cùng số mẫu."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size phải lớn hơn 0."
            )

        self.X = X
        self.y = y

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.n_samples = len(X)

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(
                self.n_samples
            )
        else:
            indices = np.arange(
                self.n_samples
            )

        X_shuffled = self.X[indices]
        y_shuffled = self.y[indices]

        for start in range(
            0,
            self.n_samples,
            self.batch_size
        ):
            end = start + self.batch_size

            if end > self.n_samples and self.drop_last:
                break

            yield (
                X_shuffled[start:end],
                y_shuffled[start:end]
            )

    def __len__(self):
        if self.drop_last:
            return (
                self.n_samples
                // self.batch_size
            )

        return int(
            np.ceil(
                self.n_samples
                / self.batch_size
            )
        )
```

Cách dùng:

```python
train_loader = DataLoader(
    X_train,
    y_train,
    batch_size=32,
    shuffle=True
)

for X_batch, y_batch in train_loader:
    ...
```

---

# 14. Training loop với mini-batch

Training loop đầy đủ:

```python
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        output = model.forward(
            X_batch,
            training=True
        )

        loss = loss_function.calculate(
            output,
            y_batch
        )

        loss_function.backward(
            output,
            y_batch
        )

        model.backward(
            loss_function.dinputs
        )

        optimizer.pre_update_params()

        for layer in trainable_layers:
            optimizer.update_params(
                layer
            )

        optimizer.post_update_params()
```

Tư duy:

```text
Mỗi batch tạo một lần gradient.

Mỗi batch tạo một lần update.

Một epoch gồm nhiều batch.

Optimizer iterations tăng theo số batch đã update.
```

---

# 15. Loss của một batch

Với batch có $m$ mẫu:

$$L_{\mathrm{batch}}=\frac{1}{m}\sum_{i=1}^{m}L_{i}$$

Ví dụ MSE:

$$L_{\mathrm{batch}}=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_{i}-y_{i})^2$$

Trong code:

```python
batch_loss = loss_function.calculate(
    y_pred_batch,
    y_batch
)
```

Nếu batch cuối có ít mẫu hơn, Loss vẫn lấy trung bình trên số mẫu của batch đó.

---

# 16. Loss của một epoch

Nếu batch size khác nhau, không nên tính epoch loss bằng trung bình đơn giản của các batch loss.

Sai:

```python
epoch_loss = np.mean(batch_losses)
```

Lý do:

```text
Batch lớn và batch nhỏ bị xem như có trọng số ngang nhau.
```

Cách đúng là tính trung bình theo số mẫu.

Nếu batch $j$ có $m_j$ mẫu và loss trung bình $L_j$, thì:

$$L_{\mathrm{epoch}}=\frac{\sum_{j=1}^{K}m_{j}L_{j}}{\sum_{j=1}^{K}m_{j}}$$

Code:

```python
epoch_loss_sum = 0.0
epoch_sample_count = 0

for X_batch, y_batch in train_loader:
    y_pred = model.forward(X_batch)

    batch_loss = loss_function.calculate(
        y_pred,
        y_batch
    )

    batch_size_actual = len(X_batch)

    epoch_loss_sum += (
        batch_loss
        * batch_size_actual
    )

    epoch_sample_count += (
        batch_size_actual
    )

epoch_loss = (
    epoch_loss_sum
    / epoch_sample_count
)
```

---

# 17. Cách thiết kế Loss tích lũy OOP

Base Loss có thể có:

```text
accumulated_sum
accumulated_count
```

Code:

```python
class Loss:
    def calculate(
        self,
        y_pred,
        y_true
    ):
        sample_losses = self.forward(
            y_pred,
            y_true
        )

        data_loss = np.mean(
            sample_losses
        )

        self.accumulated_sum += np.sum(
            sample_losses
        )

        self.accumulated_count += len(
            sample_losses
        )

        return data_loss

    def calculate_accumulated(self):
        if self.accumulated_count == 0:
            return 0.0

        return (
            self.accumulated_sum
            / self.accumulated_count
        )

    def new_pass(self):
        self.accumulated_sum = 0.0
        self.accumulated_count = 0
```

Trước mỗi epoch:

```python
loss_function.new_pass()
```

Sau mỗi batch:

```python
loss = loss_function.calculate(
    y_pred,
    y_batch
)
```

Cuối epoch:

```python
epoch_loss = (
    loss_function.calculate_accumulated()
)
```

---

# 18. Accuracy của một epoch

Accuracy cũng cần tích lũy theo số mẫu hoặc số phần tử.

Nếu batch có kích thước khác nhau, không nên lấy trung bình không trọng số của batch accuracy.

Cách đúng:

$$\mathrm{Accuracy}_{\mathrm{epoch}}=\frac{\text{tổng số dự đoán đúng}}{\text{tổng số dự đoán}}$$

Base Accuracy:

```python
class Accuracy:
    def calculate(
        self,
        predictions,
        y
    ):
        comparisons = self.compare(
            predictions,
            y
        )

        accuracy = np.mean(
            comparisons
        )

        self.accumulated_sum += np.sum(
            comparisons
        )

        self.accumulated_count += (
            comparisons.size
        )

        return accuracy

    def calculate_accumulated(self):
        if self.accumulated_count == 0:
            return 0.0

        return (
            self.accumulated_sum
            / self.accumulated_count
        )

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
```

---

# 19. Optimizer update theo batch

Trong mini-batch training, optimizer update sau mỗi batch.

Không phải sau mỗi epoch.

Sai nếu viết:

```python
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        forward()
        backward()

    optimizer.update_params(...)
```

Cách này gom gradient của nhiều batch nhưng nếu không cộng gradient đúng cách thì sai logic.

Cách phổ biến:

```python
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        forward()
        backward()
        optimizer.update_params(...)
```

Tư duy:

```text
Một batch
→ một gradient
→ một update.
```

---

# 20. Optimizer `pre_update_params()` và `post_update_params()`

Với optimizer có decay, momentum hoặc Adam, ta thường dùng:

```python
optimizer.pre_update_params()

for layer in trainable_layers:
    optimizer.update_params(layer)

optimizer.post_update_params()
```

Trong mini-batch training, đoạn này nằm trong vòng lặp batch.

Đúng:

```python
for X_batch, y_batch in train_loader:
    ...
    optimizer.pre_update_params()

    for layer in trainable_layers:
        optimizer.update_params(layer)

    optimizer.post_update_params()
```

Sai:

```python
optimizer.pre_update_params()

for X_batch, y_batch in train_loader:
    ...
    for layer in trainable_layers:
        optimizer.update_params(layer)

optimizer.post_update_params()
```

Lý do:

```text
Learning rate decay cần tính trước mỗi update.

Iteration cần tăng sau mỗi batch update.

Không phải sau mỗi epoch.
```

---

# 21. `iterations` trong optimizer là gì?

Trong optimizer:

```python
self.iterations += 1
```

`iterations` nên đại diện cho số lần update parameters.

Với mini-batch:

$$\mathrm{iterations}=\text{số batch đã update}$$

Không phải số epoch.

Nếu:

```text
epochs = 10
steps_per_epoch = 32
```

thì cuối training:

$$\mathrm{iterations}=320$$

Điều này ảnh hưởng trực tiếp đến learning rate decay:

$$\eta_{t}=\frac{\eta_{0}}{1+\mathrm{decay}\cdot t}$$

Trong đó $t$ là số lần update.

---

# 22. Mini-batch và Adam

Với Adam, mỗi batch tạo gradient mới.

Adam lưu:

```text
momentums
cache
iterations
```

Sau mỗi batch, Adam cập nhật:

$$m_{t}=\beta_{1}m_{t-1}+(1-\beta_{1})g_{t}$$

$$v_{t}=\beta_{2}v_{t-1}+(1-\beta_{2})g_{t}^{2}$$

Trong đó $g_t$ là gradient của batch hiện tại.

Sau đó bias correction:

$$\hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}$$

$$\hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}$$

Update:

$$W_{t+1}=W_{t}-\eta\frac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}}+\epsilon}$$

Ở đây $t$ là số update theo batch.

---

# 23. Batch size ảnh hưởng thế nào?

## Batch size nhỏ

Ví dụ:

```python
batch_size = 8
```

Đặc điểm:

```text
Gradient nhiễu hơn.

Update thường xuyên hơn.

Có thể generalize tốt hơn.

Training có thể dao động mạnh.

Tốn ít RAM.
```

## Batch size lớn

Ví dụ:

```python
batch_size = 512
```

Đặc điểm:

```text
Gradient ổn định hơn.

Update ít hơn trong một epoch.

Tận dụng vectorization tốt hơn.

Tốn nhiều RAM hơn.

Có thể cần learning rate lớn hơn.
```

## Batch size phổ biến

Một số giá trị thường dùng:

```text
16
32
64
128
256
```

Với bài học từ đầu bằng NumPy, có thể bắt đầu bằng:

```python
batch_size = 32
```

hoặc:

```python
batch_size = 64
```

---

# 24. Chọn batch size thế nào?

Không có một batch size đúng cho mọi bài toán.

Tư duy chọn:

```text
Dataset nhỏ:
Có thể dùng full-batch hoặc batch size nhỏ.

Dataset vừa:
Dùng 32, 64 hoặc 128.

Dataset lớn:
Dùng mini-batch để tiết kiệm RAM.

Nếu loss dao động quá mạnh:
Tăng batch size hoặc giảm learning rate.

Nếu training quá chậm:
Tăng batch size hoặc tối ưu code.

Nếu hết RAM:
Giảm batch size.
```

---

# 25. Batch size và learning rate

Batch size và learning rate liên quan với nhau.

Batch size lớn hơn thường cho gradient ổn định hơn.

Do đó đôi khi có thể dùng learning rate lớn hơn.

Batch size nhỏ làm gradient nhiễu hơn.

Do đó có thể cần learning rate nhỏ hơn để tránh dao động.

Tư duy:

```text
Batch size kiểm soát độ nhiễu của gradient.

Learning rate kiểm soát độ dài bước update.

Hai thứ này cần chọn cùng nhau.
```

---

# 26. Mini-batch và Regularization

Mini-batch training có tính nhiễu tự nhiên.

Nhiễu trong gradient đôi khi giúp mô hình không bám quá chặt vào training set.

Tuy nhiên mini-batch không thay thế regularization.

Vẫn có thể cần:

```text
L1 Regularization.

L2 Regularization.

Dropout.

Early Stopping.

Data augmentation.
```

---

# 27. Mini-batch và BatchNorm, Dropout

Một số layer phụ thuộc vào trạng thái training.

## Dropout

Trong training:

```text
Tắt ngẫu nhiên một số neuron trong mỗi batch.
```

Trong inference:

```text
Không tắt neuron.
```

Do đó `model.forward()` nên có:

```python
training=True
```

## BatchNorm

BatchNorm dùng mean và variance của batch trong training.

Trong inference, BatchNorm dùng moving average.

Do đó mini-batch ảnh hưởng trực tiếp đến BatchNorm.

Nếu batch size quá nhỏ, thống kê batch có thể nhiễu.

---

# 28. Tích hợp mini-batch vào class `Model`

Một class `Model` tốt nên hỗ trợ:

```python
model.train(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val)
)
```

Trong `train()`, model tự tạo DataLoader:

```python
train_loader = DataLoader(
    X_train,
    y_train,
    batch_size=batch_size,
    shuffle=True
)
```

Sau đó:

```python
for epoch in range(epochs):
    self.loss.new_pass()
    self.accuracy.new_pass()

    for X_batch, y_batch in train_loader:
        ...
```

---

# 29. Model train loop với mini-batch

```python
def train(
    self,
    X,
    y,
    epochs=1000,
    batch_size=32,
    print_every=100,
    validation_data=None
):
    train_loader = DataLoader(
        X,
        y,
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(epochs + 1):
        self.loss.new_pass()

        if self.accuracy is not None:
            self.accuracy.new_pass()

        for X_batch, y_batch in train_loader:
            output = self.forward(
                X_batch,
                training=True
            )

            batch_loss = self.loss.calculate(
                output,
                y_batch
            )

            if self.accuracy is not None:
                batch_accuracy = (
                    self.accuracy.calculate(
                        output,
                        y_batch
                    )
                )

            self.backward(
                output,
                y_batch
            )

            self.optimizer.pre_update_params()

            for layer in self.trainable_layers:
                self.optimizer.update_params(
                    layer
                )

            self.optimizer.post_update_params()

        epoch_loss = (
            self.loss
            .calculate_accumulated()
        )

        if self.accuracy is not None:
            epoch_accuracy = (
                self.accuracy
                .calculate_accumulated()
            )
        else:
            epoch_accuracy = None

        if epoch % print_every == 0:
            print(
                "epoch:",
                epoch,
                "loss:",
                epoch_loss,
                "accuracy:",
                epoch_accuracy
            )
```

---

# 30. Validation với mini-batch

Validation cũng có thể chạy theo batch để tránh tốn RAM.

Nhưng validation không update parameters.

```python
def evaluate(
    self,
    X,
    y,
    batch_size=32
):
    data_loader = DataLoader(
        X,
        y,
        batch_size=batch_size,
        shuffle=False
    )

    self.loss.new_pass()

    if self.accuracy is not None:
        self.accuracy.new_pass()

    for X_batch, y_batch in data_loader:
        output = self.forward(
            X_batch,
            training=False
        )

        self.loss.calculate(
            output,
            y_batch
        )

        if self.accuracy is not None:
            self.accuracy.calculate(
                output,
                y_batch
            )

    loss = self.loss.calculate_accumulated()

    if self.accuracy is not None:
        accuracy = (
            self.accuracy
            .calculate_accumulated()
        )
    else:
        accuracy = None

    return loss, accuracy
```

Tư duy:

```text
Train:
Forward, backward, update.

Validation/Test:
Chỉ forward và tính metric.
```

---

# 31. Mini-batch trong Regression

Với Regression, mỗi batch gồm nhiều cặp:

$$\{(x_{i},y_{i})\}_{i=1}^{B}$$

Model output:

$$\hat{y}_{i}=f(x_{i})$$

Loss MSE trên batch:

$$L_{\mathrm{batch}}=\frac{1}{B}\sum_{i=1}^{B}(\hat{y}_{i}-y_{i})^2$$

Trong mỗi batch:

```text
Forward.
Tính MSE.
Backward.
Update.
```

Không dùng `argmax()` trong regression.

Nếu muốn Accuracy cho regression, dùng tolerance:

$$\mathrm{Accuracy}_{\tau}=\frac{1}{B}\sum_{i=1}^{B}\mathbf{1}\bigl(|\hat{y}_{i}-y_{i}|\leq \tau\bigr)$$

---

# 32. Mini-batch trong Classification

Với Classification, mỗi batch gồm:

```text
X_batch shape: (B, n_features)

y_batch shape:
- (B,) nếu label sparse
- (B, C) nếu one-hot
```

Output model thường có shape:

```text
(B, C)
```

Prediction:

```python
predictions = np.argmax(
    output,
    axis=1
)
```

Accuracy batch:

```python
batch_accuracy = np.mean(
    predictions == y_batch
)
```

Nếu `y_batch` là one-hot:

```python
y_true = np.argmax(
    y_batch,
    axis=1
)
```

---

# 33. Mini-batch và overfitting

Mini-batch không tự động ngăn overfitting.

Nếu model quá lớn và train quá lâu, model vẫn có thể học thuộc training data.

Nên dùng validation set.

Theo dõi:

```text
Training loss.

Validation loss.
```

Dấu hiệu overfitting:

```text
Training loss tiếp tục giảm.

Validation loss bắt đầu tăng.
```

Khi đó có thể dùng:

```text
Early Stopping.

Regularization.

Dropout.

Giảm độ phức tạp model.

Tăng dữ liệu.
```

---

# 34. Early Stopping với mini-batch

Early Stopping nên dựa trên validation loss sau mỗi epoch.

Không nên dừng chỉ vì một batch có loss xấu.

Lý do:

```text
Batch loss có thể nhiễu.

Validation loss theo epoch ổn định hơn.
```

Pseudo-code:

```python
best_validation_loss = np.inf
patience = 10
wait = 0

for epoch in range(epochs):
    train_one_epoch()

    validation_loss = evaluate(
        X_val,
        y_val
    )

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        wait = 0
        save_best_parameters()
    else:
        wait += 1

    if wait >= patience:
        restore_best_parameters()
        break
```

---

# 35. Những lỗi thường gặp khi mini-batch training

## Lỗi 1: Không shuffle training data

Nếu dữ liệu có thứ tự, batch sẽ bị lệch.

Nên shuffle mỗi epoch.

---

## Lỗi 2: Tính epoch loss bằng trung bình batch loss không trọng số

Sai nếu batch cuối nhỏ hơn.

Nên tích lũy theo số mẫu.

---

## Lỗi 3: Chỉ gọi optimizer update sau epoch

Mini-batch thường update sau mỗi batch.

---

## Lỗi 4: Gọi `post_update_params()` sau mỗi layer

Sai:

```python
optimizer.pre_update_params()

optimizer.update_params(layer1)
optimizer.post_update_params()

optimizer.pre_update_params()

optimizer.update_params(layer2)
optimizer.post_update_params()
```

Đúng:

```python
optimizer.pre_update_params()

optimizer.update_params(layer1)
optimizer.update_params(layer2)

optimizer.post_update_params()
```

Trong mini-batch, đoạn đúng trên được thực hiện một lần cho mỗi batch.

---

## Lỗi 5: Quên reset Loss và Accuracy mỗi epoch

Nên gọi:

```python
loss_function.new_pass()
accuracy_function.new_pass()
```

trước mỗi epoch.

---

## Lỗi 6: Dùng `argmax()` cho Regression

Regression output shape thường là:

```text
(B, 1)
```

Nếu dùng:

```python
np.argmax(output, axis=1)
```

kết quả thường vô nghĩa.

---

## Lỗi 7: Không xử lý batch cuối

Nếu batch cuối nhỏ hơn `batch_size`, code vẫn phải chạy được.

Không nên giả định mọi batch đều có kích thước bằng nhau.

---

# 36. Checklist triển khai mini-batch

```text
Chuẩn bị X, y.

Chọn batch_size.

Tạo DataLoader.

Shuffle training data mỗi epoch.

Forward từng batch.

Tính batch loss.

Tích lũy loss theo số mẫu.

Tính batch accuracy nếu có.

Tích lũy accuracy.

Backward từng batch.

Update parameters sau từng batch.

Tăng optimizer.iterations sau từng batch.

Evaluate validation sau mỗi epoch.

Dùng early stopping nếu cần.
```

---

# 37. Tóm tắt tư duy

Mini-batch training là cách chia dataset thành nhiều phần nhỏ để train.

Tư duy cốt lõi:

```text
Một epoch:
Model đi qua toàn bộ training data một lần.

Một batch:
Một phần nhỏ của training data.

Một iteration:
Một lần update parameters.

Trong mini-batch:
Mỗi batch tạo một lần update.
```

Công thức batch loss:

$$L_{\mathrm{batch}}=\frac{1}{B}\sum_{i=1}^{B}L_{i}$$

Công thức epoch loss đúng:

$$L_{\mathrm{epoch}}=\frac{\sum_{j=1}^{K}m_{j}L_{j}}{\sum_{j=1}^{K}m_{j}}$$

Trong đó:

* $K$ là số batch.
* $m_j$ là số mẫu trong batch thứ $j$.
* $L_j$ là loss trung bình của batch thứ $j$.

Nguyên tắc quan trọng:

```text
Shuffle training data mỗi epoch.

Update optimizer sau mỗi batch.

Tích lũy loss và accuracy theo số mẫu.

Validation và test không update weights.

Optimizer iterations đếm số batch update, không đếm số epoch.

Batch size ảnh hưởng đến độ nhiễu gradient, tốc độ train và bộ nhớ.
```
