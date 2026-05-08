import numpy as np

class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        loss = np.mean(sample_losses)
        return loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_predictions, y_true):
        samples = len(y_predictions)

        y_predictions = np.clip(y_predictions, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_predictions[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_predictions * y_true, axis = 1)

        correct_confidence = -np.log(correct_confidence)
        return correct_confidence

class Loss_MeanSquareError(Loss):
    def forward(self, y_predictions, y_true):
        squared_error = (y_predictions - y_true) ** 2
        sample_losses = np.mean(squared_error, axis = 1)
        return sample_losses

