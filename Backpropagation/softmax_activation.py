import numpy as np
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dvalues):
        sum_A_dA = np.sum(self.output * dvalues, axis = 1, keepdims=True)
        self.dinputs = self.output * (dvalues - sum_A_dA) 
        return self.dinputs

softmax_output = np.array([[0.7, 0.1, 0.2]])
A = Activation_Softmax()
A.output = softmax_output
A.backward(np.array([[1.0, 0.0, 0.0]]))

print("Đầu ra Softmax hiện tại:")
print(A.output)

print("\nGradient trả về lớp trước (dinputs):")
print(A.dinputs)
