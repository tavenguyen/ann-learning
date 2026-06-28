# Losses — Code and Usage

## Loss base
```python
class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        self.output = np.mean(sample_losses)
        return self.output
```

## Activation_Softmax_Loss_CategoricalCrossEntropy
```python
class Activation_Softmax_Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Softmax activation
        exp_values = np.exp(y_pred - np.max(y_pred, axis = 1, keepdims = True))
        probs = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probs
        
        samples = len(probs)
        # Clip probabilities to prevent log(0)
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_conf = probs_clipped[range(samples), y_true]
        else:
            correct_conf = np.sum(probs_clipped * y_true, axis = 1)

        negative_log_likelihood = -np.log(correct_conf)
        return negative_log_likelihood

    def backward(self, y_pred, y_true):
        probs = self.output.copy()
        samples = len(probs)

        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        probs[range(samples), y_true] -= 1
        self.dinputs = probs / samples
        return self.dinputs
```

## Loss_MeanSquaredError
```python
class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        error = np.sum(squared_error, axis = 1)
        return error

    def backward(self, y_pred, y_true):
        samples = y_pred.shape[0] 
        outputs = y_pred.shape[1]
        self.dinputs = (2 * (y_pred - y_true)) / (outputs * samples)
        return self.dinputs
```

Notes
- Use the combined Softmax+CCE class for efficient classification training.
