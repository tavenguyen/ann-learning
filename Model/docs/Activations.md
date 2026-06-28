# Activations — Code and Usage

Extracted activation classes from Model/model.py

## Activation_Linear
```python
class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.inputs
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs
```

## Activation_ReLU
```python
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, self.inputs)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
```

## Activation_Softmax
```python
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values / (np.sum(exp_values, axis = 1, keepdims=True))
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output,single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
```

## Activation_Sigmoid
```python
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1.0 - self.output))
        return self.dinputs
```

Notes
- Prefer using Activation_Softmax_Loss_CategoricalCrossEntropy when training classification with softmax for numerical stability and efficiency.
