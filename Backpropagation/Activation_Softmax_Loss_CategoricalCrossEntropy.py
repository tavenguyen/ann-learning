import numpy as np

class Activation_Softmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.inputs = None
        self.output = None

    def forward(self, inputs, y_true):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

        y_pred = np.clip(inputs, 1e-7, 1 - 1e-7)

        # y_pred = [
        #     [0.7, 0.1, 0.2], # Mẫu 1: Mô hình đoán Lớp 0 với 70% tự tin
        #     [0.1, 0.5, 0.4], # Mẫu 2: Mô hình đoán Lớp 1 với 50% tự tin
        #     [0.02, 0.9, 0.08]  # Mẫu 3: Mô hình đoán Lớp 1 với 90% tự tin
        # ]

        # y_true_1d = [0, 1, 1]
        if(len(y_true.shape) == 1):
            confidence_score = y_pred[range(inputs), y_true]
        # y_true_2d = [
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [0, 1, 0]
        # ]
        elif(len(y_true.shape) == 2):   
            confidence_score = np.sum(y_pred * y_true, axis = 1)

        confidence_score = -np.log(confidence_score)
        return confidence_score
    
    def backward(self, dvalues, y_true):
        # dvalues là output của softmax activation, ví dụ: softmax_output = [0.7, 0.1, 0.2]
        samples = len(dvalues)

        if(len(y_true.shape) == 2):
            y_true = np.argmax(y_true, axis = 1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        
        self.dinputs /= samples
        return self.dinputs

