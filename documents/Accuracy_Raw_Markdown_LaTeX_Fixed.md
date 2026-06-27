Dưới đây là tài liệu Raw Markdown thuần, có thể copy trực tiếp vào GitHub Issues.

# [ANN] Accuracy: Bản chất, các loại Accuracy và tư duy OOP

## 1. Mục tiêu

Tài liệu này giải thích:

* Accuracy là gì.
* Bản chất toán học của Accuracy.
* Vì sao phải biến output của ANN thành prediction trước khi tính Accuracy.
* Các loại Accuracy thường gặp trong ANN.
* Sự khác nhau giữa Accuracy của Classification và Regression.
* Hạn chế của Accuracy.
* Tư duy OOP khi xây dựng các class Accuracy.
* Cách tích lũy Accuracy qua nhiều batch.

---

# 2. Accuracy là gì?

Accuracy đo tỷ lệ dự đoán đúng trên tổng số dự đoán.

Công thức cơ bản:

$$\mathrm{Accuracy}=\frac{\text{số dự đoán đúng}}{\text{tổng số dự đoán}}$$

Nếu có $N$ mẫu:

$$\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left(\hat{y}_i=y_i\right)$$

Trong đó:

* $y_i$ là nhãn thật của mẫu thứ $i$.
* $\hat{y}_i$ là nhãn dự đoán.
* $\mathbf{1}(\cdot)$ là hàm chỉ thị.

Hàm chỉ thị được hiểu là:

$$\mathbf{1}\left(\hat{y}_i=y_i\right)=\begin{cases}1, & \hat{y}_i=y_i \\ 0, & \hat{y}_i\neq y_i\end{cases}$$

Nói cách khác:

```text
Dự đoán đúng
→ nhận giá trị 1.

Dự đoán sai
→ nhận giá trị 0.

Accuracy
→ trung bình của các giá trị 0 và 1.
```

---

# 3. Ví dụ tính Accuracy

Giả sử:

```python
y_true = np.array([0, 1, 2, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0])
```

So sánh:

```python
comparisons = y_pred == y_true

print(comparisons)
```

Kết quả:

```text
[True, True, False, True, True]
```

NumPy có thể xem:

```text
True  = 1
False = 0
```

Do đó:

```python
accuracy = np.mean(
    y_pred == y_true
)
```

Ta có:

$$\mathrm{Accuracy}=\frac{4}{5}=0.8$$

Hay:

```text
Accuracy = 80%
```

---

# 4. Bản chất của việc tính Accuracy

Accuracy không được tính trực tiếp từ Loss.

Quy trình đúng là:

```text
Output của ANN
→ chuyển thành prediction
→ so sánh prediction với target
→ tính trung bình số dự đoán đúng
```

Ví dụ với Classification:

```text
Logits hoặc probabilities
→ argmax hoặc threshold
→ class prediction
→ so sánh với y_true
```

Điểm quan trọng:

```text
Loss đo mức độ sai khác.

Accuracy đo tỷ lệ quyết định đúng.
```

Hai đại lượng này có liên quan nhưng không giống nhau.

---

# 5. Loss và Accuracy khác nhau như thế nào?

Giả sử bài toán có ba class.

Hai output:

```python
output_1 = np.array([
    [0.90, 0.05, 0.05]
])

output_2 = np.array([
    [0.36, 0.34, 0.30]
])
```

Nếu class thật là class `0`, cả hai đều dự đoán:

```python
np.argmax(output, axis=1)
```

kết quả là:

```text
class 0
```

Do đó Accuracy của cả hai đều là đúng.

Tuy nhiên:

```text
Output 1:
Mô hình rất tự tin.

Output 2:
Mô hình chỉ hơn các class khác một chút.
```

Cross-Entropy Loss của hai trường hợp sẽ khác nhau.

Do đó:

```text
Accuracy chỉ quan tâm class cuối cùng đúng hay sai.

Loss còn quan tâm mức độ tự tin và khoảng cách sai lệch.
```

Optimizer thường tối ưu Loss, không trực tiếp tối ưu Accuracy.

Lý do là Accuracy sử dụng phép so sánh rời rạc, không tạo gradient hữu ích cho Backpropagation.

---

# 6. Phân loại các loại Accuracy

Accuracy cần được xây dựng theo loại bài toán.

Các nhóm thường gặp:

```text
Binary Classification Accuracy.

Categorical Accuracy.

Sparse Categorical Accuracy.

Top-k Accuracy.

Multi-label Accuracy.

Token hoặc Sequence Accuracy.

Pixel Accuracy.

Regression Tolerance Accuracy.
```

---

# 7. Binary Classification Accuracy

Binary Classification có hai class:

```text
Class 0
Class 1
```

Output thường là một xác suất:

$$0\leq p_i\leq1$$

Ta dùng threshold để chuyển xác suất thành class.

Với threshold bằng $0.5$:

$$\hat{y}_i=\begin{cases}1, & p_i\geq 0.5 \\ 0, & p_i<0.5\end{cases}$$

Code:

```python
predictions = (
    probabilities >= 0.5
).astype(int)

accuracy = np.mean(
    predictions == y_true
)
```

Ví dụ:

```python
probabilities = np.array([
    [0.8],
    [0.3],
    [0.6],
    [0.2]
])

y_true = np.array([
    [1],
    [0],
    [0],
    [0]
])
```

Prediction:

```text
[1, 0, 1, 0]
```

So với target:

```text
[1, 0, 0, 0]
```

Accuracy:

$$\mathrm{Accuracy}=\frac{3}{4}=0.75$$

---

# 8. Threshold không bắt buộc luôn bằng $0.5$

Threshold $0.5$ thường được dùng khi:

```text
Hai class có mức quan trọng tương đương.

Xác suất output đã được hiệu chỉnh tương đối tốt.
```

Nhưng có thể dùng threshold khác.

Ví dụ:

```python
threshold = 0.3

predictions = (
    probabilities >= threshold
).astype(int)
```

Giảm threshold thường làm mô hình dự đoán class dương nhiều hơn.

Điều này có thể:

```text
Tăng Recall.

Nhưng giảm Precision.
```

Do đó threshold nên được chọn bằng validation set, không nên chọn bằng test set.

---

# 9. Multi-class Categorical Accuracy

Multi-class Classification có nhiều class nhưng mỗi mẫu chỉ thuộc một class.

Ví dụ:

```text
Class 0
Class 1
Class 2
```

Output của mô hình có shape:

```text
(N, C)
```

Trong đó:

* $N$ là số mẫu.
* $C$ là số class.

Ví dụ:

```python
probabilities = np.array([
    [0.10, 0.70, 0.20],
    [0.80, 0.10, 0.10],
    [0.20, 0.30, 0.50]
])
```

Ta dùng:

```python
predictions = np.argmax(
    probabilities,
    axis=1
)
```

Kết quả:

```text
[1, 0, 2]
```

Sau đó:

```python
accuracy = np.mean(
    predictions == y_true
)
```

---

# 10. Vì sao dùng `argmax()`?

`argmax()` trả về vị trí có giá trị lớn nhất.

Với:

```python
probabilities = np.array([
    [0.10, 0.70, 0.20]
])
```

class có xác suất lớn nhất là vị trí `1`.

Do đó:

```python
prediction = np.argmax(
    probabilities,
    axis=1
)
```

trả về:

```text
[1]
```

Tư duy:

```text
Mỗi cột đại diện cho một class.

Class có score hoặc probability lớn nhất
được chọn làm prediction.
```

---

# 11. Sparse Categorical Accuracy

Sparse Categorical Accuracy dùng khi target được lưu dưới dạng class index.

Ví dụ:

```python
y_true = np.array([
    1,
    0,
    2
])
```

Prediction:

```python
predictions = np.argmax(
    probabilities,
    axis=1
)
```

Accuracy:

```python
accuracy = np.mean(
    predictions == y_true
)
```

Đây là cách phổ biến nhất khi label có dạng:

```text
0
1
2
...
C - 1
```

---

# 12. Categorical Accuracy với One-hot Target

Target cũng có thể được lưu dưới dạng one-hot.

Ví dụ:

```python
y_true = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
```

Ta cần chuyển one-hot target thành class index:

```python
true_classes = np.argmax(
    y_true,
    axis=1
)
```

Prediction:

```python
predictions = np.argmax(
    probabilities,
    axis=1
)
```

Accuracy:

```python
accuracy = np.mean(
    predictions == true_classes
)
```

Do đó class Accuracy nên kiểm tra target đang ở dạng:

```text
Sparse labels
hay
One-hot labels
```

---

# 13. Top-k Accuracy

Top-1 Accuracy yêu cầu class đúng phải đứng ở vị trí cao nhất.

Top-k Accuracy yêu cầu class đúng chỉ cần nằm trong $k$ class có score cao nhất.

Ví dụ với năm class:

```python
probabilities = np.array([
    [0.10, 0.30, 0.25, 0.20, 0.15]
])
```

Top-1 prediction là:

```text
Class 1
```

Nhưng ba class cao nhất là:

```text
Class 1
Class 2
Class 3
```

Nếu target là class `2`:

```text
Top-1:
Sai.

Top-3:
Đúng.
```

---

## Công thức Top-k Accuracy

$$\mathrm{TopKAccuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left(y_i\in\mathrm{TopK}(\hat{\mathbf{p}}_i)\right)$$

Code:

```python
top_k_indices = np.argsort(
    probabilities,
    axis=1
)[:, -k:]

comparisons = np.any(
    top_k_indices
    == y_true.reshape(-1, 1),
    axis=1
)

top_k_accuracy = np.mean(
    comparisons
)
```

Cách hiệu quả hơn:

```python
top_k_indices = np.argpartition(
    probabilities,
    -k,
    axis=1
)[:, -k:]
```

---

# 14. Khi nào dùng Top-k Accuracy?

Top-k Accuracy hữu ích khi:

```text
Có rất nhiều class.

Nhiều class tương đối giống nhau.

Muốn biết đáp án đúng có nằm trong nhóm đề xuất tốt nhất không.
```

Ví dụ:

```text
Image classification với hàng nghìn class.

Recommendation.

Information retrieval.
```

Không nên chỉ báo cáo Top-5 mà bỏ qua Top-1, vì Top-5 luôn dễ đạt cao hơn.

---

# 15. Multi-label Classification Accuracy

Trong Multi-label Classification, một mẫu có thể thuộc nhiều class cùng lúc.

Ví dụ:

```text
Một hình ảnh có thể chứa:

person
car
road
tree
```

Target:

```python
y_true = np.array([
    [1, 1, 0, 1]
])
```

Mỗi output thường dùng Sigmoid độc lập.

Prediction:

```python
predictions = (
    probabilities >= 0.5
).astype(int)
```

Có nhiều cách tính Accuracy cho Multi-label.

---

# 16. Exact Match Accuracy

Exact Match Accuracy chỉ tính đúng nếu toàn bộ label của một mẫu đều đúng.

Ví dụ:

```text
Target:
[1, 1, 0, 1]

Prediction:
[1, 1, 0, 1]
→ đúng.

Prediction:
[1, 1, 1, 1]
→ sai toàn bộ mẫu.
```

Code:

```python
sample_correct = np.all(
    predictions == y_true,
    axis=1
)

accuracy = np.mean(
    sample_correct
)
```

Công thức:

$$\mathrm{ExactMatchAccuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left(\hat{\mathbf{y}}_i=\mathbf{y}_i\right)$$

Metric này rất nghiêm khắc.

---

# 17. Label-wise Accuracy

Label-wise Accuracy tính tỷ lệ đúng trên từng phần tử nhãn.

Code:

```python
accuracy = np.mean(
    predictions == y_true
)
```

Ví dụ:

```text
Target:
[1, 1, 0, 1]

Prediction:
[1, 1, 1, 1]
```

Có ba trên bốn label đúng:

$$\mathrm{Accuracy}=\frac{3}{4}=0.75$$

Metric này dễ đạt cao hơn Exact Match Accuracy.

---

# 18. Hamming Accuracy

Hamming Accuracy trong Multi-label thường được hiểu là:

$$\mathrm{HammingAccuracy}=1-\mathrm{HammingLoss}$$

Với:

$$\mathrm{HammingLoss}=\frac{\text{số label dự đoán sai}}{\text{tổng số label}}$$

Do đó:

$$\mathrm{HammingAccuracy}=\frac{\text{số label dự đoán đúng}}{\text{tổng số label}}$$

Trong nhiều implementation, Hamming Accuracy tương đương Label-wise Accuracy.

---

# 19. Token Accuracy và Sequence Accuracy

Trong bài toán chuỗi, mỗi mẫu chứa nhiều token.

Ví dụ:

```text
Câu có 5 token.
```

## Token Accuracy

Tính tỷ lệ token dự đoán đúng:

$$\mathrm{TokenAccuracy}=\frac{\text{số token đúng}}{\text{tổng số token}}$$

## Sequence Accuracy

Chỉ tính đúng nếu toàn bộ chuỗi được dự đoán đúng:

$$\mathrm{SequenceAccuracy}=\frac{\text{số chuỗi đúng hoàn toàn}}{\text{tổng số chuỗi}}$$

Sequence Accuracy nghiêm khắc hơn Token Accuracy.

Ví dụ:

```text
Target:
A B C D

Prediction:
A B C E
```

Token Accuracy:

$$\frac{3}{4}=0.75$$

Sequence Accuracy:

$$0$$

---

# 20. Pixel Accuracy trong Image Segmentation

Trong Semantic Segmentation, mỗi pixel cần được phân class.

Pixel Accuracy:

$$\mathrm{PixelAccuracy}=\frac{\text{số pixel được phân loại đúng}}{\text{tổng số pixel}}$$

Code đơn giản:

```python
predictions = np.argmax(
    probabilities,
    axis=-1
)

pixel_accuracy = np.mean(
    predictions == masks
)
```

Hạn chế:

```text
Background thường chiếm nhiều pixel.

Mô hình có thể đạt Pixel Accuracy cao
chỉ bằng cách dự đoán background.
```

Vì vậy nên kết hợp với:

```text
IoU.

Mean IoU.

Dice Score.
```

---

# 21. Balanced Accuracy

Accuracy thông thường có thể gây hiểu nhầm khi dữ liệu mất cân bằng.

Ví dụ:

```text
Class 0: 990 mẫu.
Class 1: 10 mẫu.
```

Nếu mô hình luôn dự đoán class `0`:

$$\mathrm{Accuracy}=\frac{990}{1000}=0.99$$

Nhưng mô hình không phát hiện được bất kỳ mẫu class `1` nào.

Balanced Accuracy tính trung bình Recall của các class:

$$\mathrm{BalancedAccuracy}=\frac{1}{C}\sum_{c=1}^{C}\mathrm{Recall}_c$$

Với binary classification:

$$\mathrm{BalancedAccuracy}=\frac{\mathrm{Sensitivity}+\mathrm{Specificity}}{2}$$

Trong đó:

$$\mathrm{Sensitivity}=\frac{TP}{TP+FN}$$

$$\mathrm{Specificity}=\frac{TN}{TN+FP}$$

Balanced Accuracy phù hợp hơn khi class bị mất cân bằng.

---

# 22. Regression có Accuracy không?

Regression dự đoán giá trị liên tục.

Ví dụ:

```text
Target:
5.0

Prediction:
5.01
```

Nếu dùng:

```python
prediction == target
```

kết quả là `False`, dù dự đoán rất tốt.

Do đó Regression không dùng Accuracy tuyệt đối theo cách Classification.

Regression thường dùng:

```text
MAE.

MSE.

RMSE.

R².
```

---

# 23. Tolerance Accuracy cho Regression

Có thể tự định nghĩa một prediction là đúng nếu sai số nhỏ hơn một ngưỡng.

Với tolerance $\tau$:

$$\mathrm{Accuracy}_{\tau}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left(\left|\hat{y}_i-y_i\right|\leq\tau\right)$$

Code:

```python
tolerance = 2.0

comparisons = (
    np.abs(y_pred - y_true)
    <= tolerance
)

accuracy = np.mean(
    comparisons
)
```

Nên đặt tên rõ:

```text
Accuracy@2.0
```

Nó có nghĩa:

```text
Tỷ lệ prediction có sai số không vượt quá 2 đơn vị.
```

Đây không phải metric chuẩn chung vì kết quả phụ thuộc mạnh vào tolerance.

---

# 24. Relative Tolerance Accuracy

Khi target có scale rất khác nhau, ngưỡng tuyệt đối có thể không phù hợp.

Có thể dùng sai số tương đối:

$$\frac{|\hat{y}_i-y_i|}{|y_i|+\epsilon}\leq\tau$$

Code:

```python
epsilon = 1e-7
relative_tolerance = 0.05

relative_errors = (
    np.abs(y_pred - y_true)
    / (
        np.abs(y_true)
        + epsilon
    )
)

accuracy = np.mean(
    relative_errors
    <= relative_tolerance
)
```

Ví dụ:

```text
relative_tolerance = 0.05
```

có nghĩa prediction được tính đúng nếu sai số tương đối không quá 5%.

---

# 25. Accuracy không phải lúc nào cũng là metric tốt

Accuracy có thể gây hiểu nhầm khi:

```text
Dữ liệu mất cân bằng.

Chi phí của các loại sai lầm khác nhau.

Cần quan tâm class hiếm.

Dự đoán xác suất cần được hiệu chỉnh.

Bài toán là Multi-label hoặc Regression.
```

Nên cân nhắc thêm:

```text
Precision.

Recall.

F1-score.

ROC-AUC.

PR-AUC.

Confusion Matrix.
```

---

# 26. Tư duy OOP khi xây dựng Accuracy

Có thể tách thành:

```text
Base class Accuracy.

Các class con triển khai compare().
```

Base class chịu trách nhiệm:

```text
Tính Accuracy của batch.

Tích lũy kết quả qua nhiều batch.

Reset trạng thái cho epoch mới.
```

Class con chịu trách nhiệm:

```text
Chuyển prediction và target
thành mảng True hoặc False.
```

Tư duy:

```text
Base class biết cách tính trung bình.

Class con biết thế nào được xem là đúng.
```

---

# 27. Base class Accuracy

```python
import numpy as np

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

    def compare(
        self,
        predictions,
        y
    ):
        raise NotImplementedError(
            "Class con phải triển khai compare()."
        )
```

Trước mỗi epoch:

```python
accuracy.new_pass()
```

Sau mỗi batch:

```python
batch_accuracy = accuracy.calculate(
    predictions,
    y_batch
)
```

Cuối epoch:

```python
epoch_accuracy = (
    accuracy.calculate_accumulated()
)
```

---

# 28. Vì sao cần tích lũy Accuracy?

Giả sử có hai batch:

```text
Batch 1:
90 mẫu, Accuracy = 90%.

Batch 2:
10 mẫu, Accuracy = 50%.
```

Không nên tính trung bình đơn giản:

$$\frac{0.9+0.5}{2}=0.7$$

vì hai batch có số mẫu khác nhau.

Cách đúng:

$$\mathrm{Accuracy}=\frac{90\cdot0.9+10\cdot0.5}{100}$$

$$\mathrm{Accuracy}=\frac{81+5}{100}=0.86$$

Do đó base class tích lũy:

```text
Tổng số dự đoán đúng.

Tổng số phần tử đã đánh giá.
```

---

# 29. Class Binary Accuracy

```python
class Accuracy_Binary(Accuracy):

    def __init__(
        self,
        threshold=0.5
    ):
        if not 0 <= threshold <= 1:
            raise ValueError(
                "threshold phải nằm trong [0, 1]."
            )

        self.threshold = threshold
        self.new_pass()

    def compare(
        self,
        predictions,
        y
    ):
        predicted_classes = (
            predictions
            >= self.threshold
        ).astype(int)

        return (
            predicted_classes
            == y
        )
```

Cách dùng:

```python
accuracy = Accuracy_Binary(
    threshold=0.5
)

batch_accuracy = accuracy.calculate(
    probabilities,
    y_true
)
```

---

# 30. Class Categorical Accuracy

```python
class Accuracy_Categorical(Accuracy):

    def __init__(self):
        self.new_pass()

    def compare(
        self,
        predictions,
        y
    ):
        if predictions.ndim == 2:
            predicted_classes = np.argmax(
                predictions,
                axis=1
            )
        else:
            predicted_classes = predictions

        if y.ndim == 2:
            true_classes = np.argmax(
                y,
                axis=1
            )
        else:
            true_classes = y

        return (
            predicted_classes
            == true_classes
        )
```

Class này hỗ trợ:

```text
Prediction là probabilities hoặc class index.

Target là sparse labels hoặc one-hot labels.
```

---

# 31. Class Top-k Accuracy

```python
class Accuracy_TopK(Accuracy):

    def __init__(self, k=5):
        if k < 1:
            raise ValueError(
                "k phải lớn hơn hoặc bằng 1."
            )

        self.k = k
        self.new_pass()

    def compare(
        self,
        predictions,
        y
    ):
        if predictions.ndim != 2:
            raise ValueError(
                "Top-k Accuracy yêu cầu predictions có shape (N, C)."
            )

        if self.k > predictions.shape[1]:
            raise ValueError(
                "k không được lớn hơn số class."
            )

        if y.ndim == 2:
            true_classes = np.argmax(
                y,
                axis=1
            )
        else:
            true_classes = y

        top_k_indices = np.argpartition(
            predictions,
            -self.k,
            axis=1
        )[:, -self.k:]

        return np.any(
            top_k_indices
            == true_classes.reshape(-1, 1),
            axis=1
        )
```

Cách dùng:

```python
top5_accuracy = Accuracy_TopK(
    k=5
)
```

---

# 32. Class Multi-label Exact Match Accuracy

```python
class Accuracy_Multilabel_ExactMatch(
    Accuracy
):

    def __init__(
        self,
        threshold=0.5
    ):
        self.threshold = threshold
        self.new_pass()

    def compare(
        self,
        predictions,
        y
    ):
        predicted_labels = (
            predictions
            >= self.threshold
        ).astype(int)

        return np.all(
            predicted_labels == y,
            axis=1
        )
```

Mỗi mẫu chỉ được tính đúng nếu toàn bộ label đúng.

---

# 33. Class Multi-label Label-wise Accuracy

```python
class Accuracy_Multilabel_LabelWise(
    Accuracy
):

    def __init__(
        self,
        threshold=0.5
    ):
        self.threshold = threshold
        self.new_pass()

    def compare(
        self,
        predictions,
        y
    ):
        predicted_labels = (
            predictions
            >= self.threshold
        ).astype(int)

        return (
            predicted_labels
            == y
        )
```

Vì output của `compare()` có shape:

```text
(N, C)
```

base class sẽ tính trung bình trên toàn bộ label.

---

# 34. Class Regression Tolerance Accuracy

```python
class Accuracy_RegressionTolerance(
    Accuracy
):

    def __init__(
        self,
        tolerance=None
    ):
        self.tolerance = tolerance
        self.new_pass()

    def init(self, y):
        if self.tolerance is None:
            self.tolerance = (
                np.std(y)
                / 250
            )

    def compare(
        self,
        predictions,
        y
    ):
        if self.tolerance is None:
            raise RuntimeError(
                "Cần gọi init(y) trước khi tính Accuracy."
            )

        return (
            np.abs(
                predictions - y
            )
            <= self.tolerance
        )
```

Cách dùng:

```python
accuracy = Accuracy_RegressionTolerance(
    tolerance=2.0
)

accuracy_value = accuracy.calculate(
    y_pred,
    y_true
)
```

Hoặc tự xác định tolerance từ dữ liệu:

```python
accuracy = Accuracy_RegressionTolerance()

accuracy.init(y_train)
```

Cần nhớ:

```text
Tolerance phải có ý nghĩa thực tế.

Không nên chọn tolerance chỉ để Accuracy trông cao.
```

---

# 35. Class Relative Regression Accuracy

```python
class Accuracy_RegressionRelative(
    Accuracy
):

    def __init__(
        self,
        relative_tolerance=0.05,
        epsilon=1e-7
    ):
        if relative_tolerance < 0:
            raise ValueError(
                "relative_tolerance không được âm."
            )

        self.relative_tolerance = (
            relative_tolerance
        )

        self.epsilon = epsilon
        self.new_pass()

    def compare(
        self,
        predictions,
        y
    ):
        relative_error = (
            np.abs(
                predictions - y
            )
            / (
                np.abs(y)
                + self.epsilon
            )
        )

        return (
            relative_error
            <= self.relative_tolerance
        )
```

---

# 36. Cách dùng Accuracy trong training loop

Ví dụ Categorical Classification:

```python
accuracy_function = (
    Accuracy_Categorical()
)

for epoch in range(epochs):
    accuracy_function.new_pass()

    for X_batch, y_batch in batches:
        model.forward(X_batch)

        predictions = np.argmax(
            output_layer.output,
            axis=1
        )

        batch_accuracy = (
            accuracy_function.calculate(
                predictions,
                y_batch
            )
        )

    epoch_accuracy = (
        accuracy_function
        .calculate_accumulated()
    )

    print(
        f"epoch={epoch}, "
        f"accuracy={epoch_accuracy:.4f}"
    )
```

Class `Accuracy_Categorical` cũng có thể nhận trực tiếp probabilities:

```python
batch_accuracy = (
    accuracy_function.calculate(
        output_layer.output,
        y_batch
    )
)
```

---

# 37. Thiết kế OOP tổng quát hơn

Có thể xây dựng factory đơn giản:

```python
def create_accuracy(
    task_type,
    **kwargs
):
    if task_type == "binary":
        return Accuracy_Binary(
            **kwargs
        )

    if task_type == "categorical":
        return Accuracy_Categorical()

    if task_type == "top_k":
        return Accuracy_TopK(
            **kwargs
        )

    if task_type == "multilabel_exact":
        return (
            Accuracy_Multilabel_ExactMatch(
                **kwargs
            )
        )

    if task_type == "multilabel_labelwise":
        return (
            Accuracy_Multilabel_LabelWise(
                **kwargs
            )
        )

    if task_type == "regression":
        return (
            Accuracy_RegressionTolerance(
                **kwargs
            )
        )

    raise ValueError(
        f"Không hỗ trợ task_type={task_type}"
    )
```

Cách dùng:

```python
accuracy = create_accuracy(
    task_type="binary",
    threshold=0.5
)
```

---

# 38. Phân chia trách nhiệm giữa Model, Loss và Accuracy

## Model

```text
Tạo output hoặc prediction score.
```

## Activation output

```text
Biến logits thành probabilities
hoặc giữ nguyên giá trị regression.
```

## Loss

```text
Đo sai số để Backpropagation.

Cần có gradient.
```

## Accuracy

```text
Đánh giá prediction đúng hay sai.

Không tham gia Backpropagation.

Không cập nhật weights.
```

Tư duy đúng:

```text
Loss dùng để học.

Accuracy dùng để đánh giá.
```

---

# 39. Accuracy có cần backward không?

Không.

Accuracy thường không có:

```python
backward()
```

Lý do:

* Accuracy không được dùng để tính gradient.
* Accuracy chứa các phép toán rời rạc như `argmax`, threshold và equality.
* Các phép toán này không khả vi theo cách hữu ích cho Gradient Descent.

Do đó class Accuracy chỉ cần:

```text
compare()

calculate()

calculate_accumulated()

new_pass()
```

---

# 40. Có nên lưu Accuracy trong Model không?

Có thể để Model quản lý:

```text
Loss object.

Accuracy object.

Optimizer object.
```

Ví dụ:

```python
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(),
    accuracy=Accuracy_Categorical()
)
```

Nhưng từng class vẫn nên có trách nhiệm riêng.

Không nên viết toàn bộ logic Accuracy trực tiếp trong Dense Layer hoặc Optimizer.

---

# 41. Những lỗi thường gặp khi tính Accuracy

## Dùng `argmax()` cho Regression

Sai:

```python
predictions = np.argmax(
    regression_output,
    axis=1
)
```

Nếu output có shape `(N, 1)`, kết quả luôn là `0`.

---

## Dùng Loss output làm prediction

Sai:

```python
predictions = np.argmax(
    loss_function.output,
    axis=1
)
```

Prediction phải lấy từ output của model, không phải Loss.

---

## So sánh số thực bằng `==`

Sai trong Regression:

```python
accuracy = np.mean(
    y_pred == y_true
)
```

Nên dùng MAE, RMSE, $R^2$, hoặc tolerance nếu có ý nghĩa.

---

## Không xử lý one-hot labels

Nếu target có shape `(N, C)`, cần chuyển bằng:

```python
y_true = np.argmax(
    y_true,
    axis=1
)
```

trước khi so sánh class index.

---

## Trung bình Accuracy của các batch không cùng kích thước

Không nên lấy trung bình không trọng số của các batch Accuracy.

Nên tích lũy tổng số prediction đúng và tổng số mẫu.

---

## Dùng Accuracy duy nhất cho dữ liệu mất cân bằng

Nên bổ sung:

```text
Balanced Accuracy.

Precision.

Recall.

F1-score.

Confusion Matrix.
```

---

# 42. Bảng phân biệt các loại Accuracy

| Loại                        | Bài toán                    | Cách chuyển prediction    | Điều kiện đúng                |
| --------------------------- | --------------------------- | ------------------------- | ----------------------------- |
| Binary Accuracy             | Binary Classification       | Threshold                 | Class dự đoán bằng class thật |
| Categorical Accuracy        | Multi-class                 | `argmax()`                | Class lớn nhất đúng           |
| Sparse Categorical Accuracy | Multi-class với label index | `argmax()`                | Class index đúng              |
| Top-k Accuracy              | Nhiều class                 | Chọn $k$ score lớn nhất   | Target nằm trong Top-$k$      |
| Exact Match Accuracy        | Multi-label                 | Threshold từng label      | Toàn bộ label đúng            |
| Label-wise Accuracy         | Multi-label                 | Threshold từng label      | Từng label được so sánh riêng |
| Token Accuracy              | Sequence                    | `argmax()` theo token     | Token đúng                    |
| Sequence Accuracy           | Sequence                    | So sánh toàn chuỗi        | Toàn bộ chuỗi đúng            |
| Pixel Accuracy              | Segmentation                | `argmax()` theo pixel     | Pixel đúng class              |
| Tolerance Accuracy          | Regression                  | So sánh sai số với ngưỡng | Sai số nhỏ hơn tolerance      |

---

# 43. Tóm tắt tư duy

Accuracy có bản chất:

$$\mathrm{Accuracy}=\mathrm{mean}(\mathrm{comparison})$$

Trong đó `comparison` là mảng Boolean:

```text
True:
Dự đoán được xem là đúng.

False:
Dự đoán được xem là sai.
```

Điểm quan trọng nhất là:

```text
Mỗi bài toán có một định nghĩa "đúng" khác nhau.
```

Binary Classification:

```text
Dùng threshold.
```

Multi-class Classification:

```text
Dùng argmax.
```

Multi-label Classification:

```text
Dùng threshold cho từng label.
```

Regression:

```text
Không dùng equality.

Ưu tiên MAE, RMSE và R².

Chỉ dùng tolerance accuracy khi ngưỡng có ý nghĩa.
```

Tư duy OOP:

```text
Base Accuracy:
- Tính trung bình.
- Tích lũy qua batch.
- Reset mỗi epoch.

Class con:
- Định nghĩa compare().
- Xác định prediction nào được xem là đúng.
```

Nguyên tắc cuối cùng:

```text
Loss định nghĩa mô hình cần tối ưu điều gì.

Accuracy định nghĩa cách con người đánh giá kết quả.

Accuracy không thay thế Loss.

Accuracy không tham gia Backpropagation.
```
