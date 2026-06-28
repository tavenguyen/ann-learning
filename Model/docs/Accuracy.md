# Accuracy — Code and Usage

## Accuracy base
```python
class Accuracy:
    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, y_pred, y_true):
        comparisions = self.compare(y_pred, y_true)
        accuracy = np.mean(comparisions)
        
        self.accumulated_sum += np.sum(comparisions)
        self.accumulated_count += comparisions.size

        return accuracy

    def calculate_accumulated(self):
        if self.accumulated_count == 0:
            return 0.0
        
        return (self.accumulated_sum / self.accumulated_count)

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
```

## Accuracy_RegressionTolerance
```python
class Accuracy_RegressionTolerance(Accuracy):
    def __init__(self, tolerance=None):
        self.tolerance = tolerance
        self.new_pass()

    def init(self, y):
        if self.tolerance is None:
            self.tolerance = (np.std(y)/ 250)

    def compare(self, predictions, y):
        if self.tolerance is None:
            raise RuntimeError(
                "Cần gọi init(y) trước khi tính Accuracy."
            )

        return (np.abs(predictions - y) <= self.tolerance)
```

## Accuracy_CategoricalClassification
```python
class Accuracy_CategoricalClassification(Accuracy):
    def compare(self, predictions, y_true):
        if predictions.ndim == 2:
            predicted_classes = np.argmax(
                predictions,
                axis=1
            )
        else:
            predicted_classes = predictions

        if y_true.ndim == 2:
            true_classes = np.argmax(
                y_true,
                axis=1
            )
        else:
            true_classes = y_true

        return (
            predicted_classes
            == true_classes
        )
```

Usage notes
- Call accuracy.new_pass() before an epoch, and accuracy.calculate on each batch to accumulate.
