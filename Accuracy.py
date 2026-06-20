import numpy as np
class Accuracy:
    def calculate(self, y_pred, y_true):
        comparisions = (y_true == y_pred)
        accuracy = np.mean(comparisions)
        return accuracy

class Accuracy_Categorical:
    def calculate(self, y_pred, y_true):
        # mô hình có shape (N, C) với N là số lượng mẫu, C là số classes
        predictions = np.argmax(y_pred, axis = 1)

        # xử lý one-hot target
        # y_true = np.array([
        #     [0, 1, 0],
        #     [1, 0, 0],
        #     [0, 0, 1]
        # ])
        if y_true.shape() == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        comparisions = (y_true == predictions)
        return np.mean(comparisions)

# Không thể viết accuracy cho Regression theo cách Classification 
# Vì target = 5.0 nhưng prediction ra 5.01, tuy kết quả dự đoán tốt nhưng kết quả comparisions vẫn là false.
# => Đặt một tolerance accuracy cho regression, nếu nó nằm trong ngưỡng thì là đúng
class Accuracy_ToleranceRegression:
    def calculate(self, y_pred, y_true, tolerance = 2.0):
        comparisions = (np.abs(y_pred - y_true) <= tolerance)
        accuracy = np.mean(comparisions) 
        return accuracy